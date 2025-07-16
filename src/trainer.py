import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from src.config import DEVICE, LEARNING_RATE, EPOCHS, OUTPUT_DIR
from src.data_loader import get_dataloader
from src.model import get_model_for_finetuning
from src.utils import ensure_dir


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(dataloader, desc="Training Epoch"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    avg_loss = running_loss / len(dataloader.dataset)
    return avg_loss


def run_finetuning():
    print("Starting fine-tuning process...")
    print(f"Using device: {DEVICE}")

    # 1. Load data
    train_loader, num_classes, _ = get_dataloader(is_train=True)

    # 2. Model
    model = get_model_for_finetuning(num_classes)

    # 3. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )

    # 4. Training loop
    checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    ensure_dir(checkpoints_dir)
    best_model_path = os.path.join(checkpoints_dir, "best_model.pth")
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        print(f"Training Loss: {train_loss:.4f}")

        # Save per-epoch checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Update best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), best_model_path)
            print(f" New best model saved: {best_model_path}")

    print("\n Fine-tuning complete.")
    return best_model_path
