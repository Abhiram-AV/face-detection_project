import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from tqdm import tqdm
from torchvision import models

from src.config import (
    PRETRAINED_MODEL_NAME, GPU_ID, IMG_SIZE, INSIGHTFACE_DOWNLOAD_PATH,
    DEVICE, INSIGHTFACE_DET_THRESHOLD
)

# Inference Model (ONNX-based)
class FaceRecognitionModel:
    def __init__(self, model_name=PRETRAINED_MODEL_NAME):
        print(f"Initializing FaceAnalysis app with model: {model_name} on ctx_id={GPU_ID}")
        self.app = FaceAnalysis(
            name=model_name,
            root=INSIGHTFACE_DOWNLOAD_PATH,
            allowed_modules=['detection', 'landmark_2d_106', 'recognition']
        )
        self.app.prepare(ctx_id=GPU_ID)

        # Load landmark model
        self.landmark_model = None
        for model in self.app.models:
            if hasattr(model, 'taskname') and model.taskname == 'landmark_2d_106':
                self.landmark_model = model
                break

        if self.landmark_model is None:
            try:
                temp_model = insightface.model_zoo.get_model('2d106det')
                providers_list = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
                temp_model.prepare(ctx_id=GPU_ID, providers=providers_list)
                self.landmark_model = temp_model
                print("Loaded '2d106det' (landmark_2d_106) via model_zoo fallback.")
            except Exception as e:
                print(f"Error loading landmark model: {e}")

        if self.landmark_model is None:
            raise RuntimeError("'landmark_2d_106' model not found. Ensure InsightFace is properly set up.")

    def get_embeddings(self, image_paths):
        embeddings = []
        valid_paths_indices = []

        for i, path in enumerate(tqdm(image_paths, desc="Extracting embeddings")):
            img = cv2.imread(path)
            if img is None or img.shape[0] == 0 or img.shape[1] == 0:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.app.get(img_rgb)

            if faces:
                main_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                embeddings.append(main_face.normed_embedding)
                valid_paths_indices.append(i)

        return np.array(embeddings), valid_paths_indices

    def get_landmarks_from_image(self, img_rgb):
        if self.landmark_model is None or not hasattr(self.landmark_model, 'session'):
            return None

        try:
            input_size = self.landmark_model.input_shape[2]
            img_resized = cv2.resize(img_rgb, (input_size, input_size))
            input_data = img_resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data.transpose(2, 0, 1), axis=0)

            output = self.landmark_model.session.run(
                self.landmark_model.output_names,
                {self.landmark_model.input_names[0]: input_data}
            )[0][0]

            landmarks = output.reshape((-1, 2))
            scale_x = img_rgb.shape[1] / input_size
            scale_y = img_rgb.shape[0] / input_size
            landmarks[:, 0] *= scale_x
            landmarks[:, 1] *= scale_y
            return landmarks
        except Exception as e:
            print(f"Landmark extraction failed: {e}")
            return None


# Fine-tuning Model (PyTorch)
class ArcFaceBackbone(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)  # Use resnet50 for better performance
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)  # L2 normalize embeddings
        return x


def get_model_for_finetuning(num_classes):
    """
    Returns a trainable PyTorch model: ResNet-based backbone + classifier.
    """
    print("Initializing fine-tuning model using PyTorch ResNet backbone...")
    embedding_size = 512

    backbone = ArcFaceBackbone(embedding_size=embedding_size)
    classifier = nn.Linear(embedding_size, num_classes)

    model = nn.Sequential(backbone, classifier)
    model.to(torch.device(DEVICE))

    return model
