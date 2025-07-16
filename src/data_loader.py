import os
import json
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.config import PROCESSED_IMAGES_DIR, DATA_SPLIT_FILE, BATCH_SIZE


def get_identity_filepaths(base_dir, identities):
    """Gets all file paths and their corresponding labels for a list of identities."""
    filepaths = []
    labels = []

    identity_to_label = {identity: i for i, identity in enumerate(identities)}

    for identity in tqdm(identities, desc="Loading file paths"):
        identity_path = os.path.join(base_dir, identity)
        for root, _, files in os.walk(identity_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(root, file))
                    labels.append(identity_to_label[identity])

    return filepaths, labels, identity_to_label


class FaceDataset(Dataset):
    """Custom PyTorch Dataset for face images."""
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def identity(x):
    """Identity transform (used to avoid lambda)."""
    return x


def get_dataloader(is_train=True):
    """Creates a DataLoader for training or evaluation using processed images."""
    with open(DATA_SPLIT_FILE, 'r') as f:
        split = json.load(f)

    identities = split['train'] if is_train else split['eval']
    filepaths, labels, identity_to_label_map = get_identity_filepaths(PROCESSED_IMAGES_DIR, identities)

    if is_train:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.Lambda(identity),  # identity instead of lambda x: x
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    # On Windows, num_workers > 0 + lambda = crash, so this version is safe
    dataloader = DataLoader(dataset=FaceDataset(filepaths, labels, transform),
                            batch_size=BATCH_SIZE,
                            shuffle=is_train,
                            num_workers=0)  # Set to 0 for safety on Windows

    return dataloader, len(identities), identity_to_label_map
