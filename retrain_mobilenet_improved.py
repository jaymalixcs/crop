"""
Improved MobileNetV2 Fine-Tuning Script for Crop Disease Classification
Designed for Domain Generalization (Real-World Field Images)

Key Enhancements:
1. Field Data Augmentation: ColorJitter, RandomResizedCrop, GaussianBlur, Cutout (RandomErasing)
2. Fine-Tuning: Unfreezes upper feature layers (features 14-18) in Stage 2 with lower learning rate.
3. Cosine Annealing Learning Rate Scheduler.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# Target 15 Classes
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

def build_improved_transforms(img_size=224):
    """
    Robust augmentations to bridge lab-to-field domain gap.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value=0)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def create_model(num_classes=15):
    """
    MobileNetV2 with custom linear classifier head.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

def train_model(data_dir, output_path="multicrop_classifier_improved.pt", epochs_stage1=5, epochs_stage2=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")

    train_tf, val_tf = build_improved_transforms()

    if not os.path.exists(data_dir):
        print(f"[!] Dataset directory not found: {data_dir}")
        return

    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = create_model(len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()

    # === STAGE 1: Freeze base model, train head only ===
    print("\n[=== STAGE 1: Training Classifier Head ===]")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    for epoch in range(epochs_stage1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / train_size
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / val_size
        print(f"Stage 1 - Epoch {epoch+1}/{epochs_stage1} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.2%}")

    # === STAGE 2: Unfreeze upper feature layers (fine-tuning) ===
    print("\n[=== STAGE 2: Fine-Tuning Upper MobileNet Layers (14-18) ===]")
    # Unfreeze features from index 14 onwards
    for idx, child in enumerate(model.features.children()):
        if idx >= 14:
            for param in child.parameters():
                param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)

    best_val_acc = 0.0

    for epoch in range(epochs_stage2):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / train_size

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / val_size

        print(f"Stage 2 - Epoch {epoch+1}/{epochs_stage2} - train_loss: {train_loss:.4f} - val_acc: {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model_state': model.state_dict(), 'class_names': CLASS_NAMES}, output_path)
            print(f"  --> Saved new best checkpoint to {output_path} (Acc: {best_val_acc:.2%})")

    print(f"\n[✓] Retraining complete. Best validation accuracy: {best_val_acc:.2%}")

if __name__ == "__main__":
    print("Script ready for Kaggle/Colab execution.")
