import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class RunwayDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, split='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.split = split
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        
        # Filter to ensure we have matching pairs
        self.valid_files = []
        for img_file in self.image_files:
            mask_file = img_file  # Assuming same filename
            if mask_file in self.mask_files:
                self.valid_files.append(img_file)
        
        print(f"Found {len(self.valid_files)} valid image-mask pairs for {split} split")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        img_name = self.valid_files[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, img_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Normalize mask to 0-1
        mask = mask / 255.0
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to tensor and add channel dimension
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image, mask

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part
        for feature in features:
            self.downs.append(self.double_conv(in_channels, feature))
            in_channels = feature
        
        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(self.double_conv(feature*2, feature))
        
        self.bottleneck = self.double_conv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        # Down
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Up
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = F.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return torch.sigmoid(self.final_conv(x))
    
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def dice_loss(pred, target, smooth=1e-5):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def bce_dice_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice

def calculate_metrics(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    target_binary = target.float()
    
    # Flatten for metric calculation
    pred_flat = pred_binary.cpu().numpy().flatten()
    target_flat = target_binary.cpu().numpy().flatten()
    
    # Calculate metrics
    iou = jaccard_score(target_flat, pred_flat, average='binary')
    precision = precision_score(target_flat, pred_flat, average='binary', zero_division=0)
    recall = recall_score(target_flat, pred_flat, average='binary', zero_division=0)
    f1 = f1_score(target_flat, pred_flat, average='binary', zero_division=0)
    
    return iou, precision, recall, f1

def train_model():
    # Data paths
    base_path = "../archive/1920x1080/1920x1080"
    labels_path = "../archive/labels/labels/areas"
    
    # Training data
    train_image_dir = os.path.join(base_path, "split_70_10_20", "train")
    train_mask_dir = os.path.join(labels_path, "train_labels_1920x1080")
    
    # Validation data
    val_image_dir = os.path.join(base_path, "split_70_10_20", "val")
    val_mask_dir = os.path.join(labels_path, "test_labels_1920x1080")  # Using test labels for validation
    
    # Test data
    test_image_dir = os.path.join(base_path, "split_70_10_20", "test")
    test_mask_dir = os.path.join(labels_path, "test_labels_1920x1080")
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Create datasets
    train_dataset = RunwayDataset(train_image_dir, train_mask_dir, train_transform, 'train')
    val_dataset = RunwayDataset(val_image_dir, val_mask_dir, val_transform, 'val')
    test_dataset = RunwayDataset(test_image_dir, test_mask_dir, val_transform, 'test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Loss and optimizer
    criterion = bce_dice_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training parameters
    num_epochs = 50
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_metrics = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_f1 = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate metrics
                iou, _, _, f1 = calculate_metrics(outputs, masks)
                val_iou += iou
                val_f1 += f1
        
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_f1 /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append({'iou': val_iou, 'f1': val_f1})
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val IoU: {val_iou:.4f}')
        print(f'  Val F1: {val_f1:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_runway_model.pth')
            print(f'  New best model saved! (Val Loss: {val_loss:.4f})')
        
        print('-' * 50)
    
    # Test evaluation
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('best_runway_model.pth'))
    model.eval()
    
    test_loss = 0.0
    test_iou = 0.0
    test_precision = 0.0
    test_recall = 0.0
    test_f1 = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            iou, precision, recall, f1 = calculate_metrics(outputs, masks)
            test_iou += iou
            test_precision += precision
            test_recall += recall
            test_f1 += f1
    
    test_loss /= len(test_loader)
    test_iou /= len(test_loader)
    test_precision /= len(test_loader)
    test_recall /= len(test_loader)
    test_f1 /= len(test_loader)
    
    print(f'\nFinal Test Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test IoU: {test_iou:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1: {test_f1:.4f}')
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot([m['iou'] for m in val_metrics], label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Validation IoU')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot([m['f1'] for m in val_metrics], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, test_iou, test_f1

if __name__ == "__main__":
    model, test_iou, test_f1 = train_model()
    print(f"\n🎯 Final Model Performance:")
    print(f"   IoU Score: {test_iou:.4f}")
    print(f"   F1 Score: {test_f1:.4f}")
    
    if test_iou > 0.7:
        print("🏆 Excellent performance! This should give you a very good score.")
    elif test_iou > 0.5:
        print("✅ Good performance! This should give you a decent score.")
    else:
        print("⚠️  Performance could be improved. Consider longer training or different architecture.")

