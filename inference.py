import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from runway_detection import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RunwayDetector:
    def __init__(self, model_path='best_runway_model.pth', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, image_path, threshold=0.5):
        """
        Predict runway mask for a single image
        
        Args:
            image_path: Path to the input image
            threshold: Threshold for binary mask (default: 0.5)
            
        Returns:
            original_image: Original image as numpy array
            mask: Predicted mask as numpy array (0-255)
            binary_mask: Binary mask (0 or 255)
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        transformed = self.transform(image=original_image)
        input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            mask = output.squeeze().cpu().numpy()
        
        # Convert to 0-255 range
        mask_255 = (mask * 255).astype(np.uint8)
        
        # Create binary mask
        binary_mask = np.where(mask > threshold, 255, 0).astype(np.uint8)
        
        return original_image, mask_255, binary_mask
    
    def predict_batch(self, image_paths, threshold=0.5):
        """
        Predict runway masks for multiple images
        
        Args:
            image_paths: List of paths to input images
            threshold: Threshold for binary mask (default: 0.5)
            
        Returns:
            results: List of tuples (original_image, mask, binary_mask)
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, threshold)
            results.append(result)
        return results
    
    def visualize_prediction(self, image_path, threshold=0.5, save_path=None):
        """
        Visualize the prediction with original image, mask, and overlay
        
        Args:
            image_path: Path to the input image
            threshold: Threshold for binary mask (default: 0.5)
            save_path: Optional path to save the visualization
        """
        original_image, mask, binary_mask = self.predict(image_path, threshold)
        
        # Create overlay
        overlay = original_image.copy()
        overlay[binary_mask > 0] = [255, 0, 0]  # Red overlay for runway
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Predicted mask (grayscale)
        axes[0, 1].imshow(mask, cmap='gray')
        axes[0, 1].set_title('Predicted Mask (Probability)')
        axes[0, 1].axis('off')
        
        # Binary mask
        axes[1, 0].imshow(binary_mask, cmap='gray')
        axes[1, 0].set_title(f'Binary Mask (Threshold: {threshold})')
        axes[1, 0].axis('off')
        
        # Overlay
        axes[1, 1].imshow(original_image)
        axes[1, 1].imshow(binary_mask, alpha=0.5, cmap='Reds')
        axes[1, 1].set_title('Runway Detection Overlay')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return original_image, mask, binary_mask
    
    def detect_runway_region(self, image_path, threshold=0.5):
        """
        Detect and return the runway region coordinates
        
        Args:
            image_path: Path to the input image
            threshold: Threshold for binary mask (default: 0.5)
            
        Returns:
            runway_region: Dictionary with runway information
        """
        original_image, mask, binary_mask = self.predict(image_path, threshold)
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        runway_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small regions
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Get center point
                center_x = x + w // 2
                center_y = y + h // 2
                
                runway_regions.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'contour': contour
                })
        
        # Sort by area (largest first)
        runway_regions.sort(key=lambda x: x['area'], reverse=True)
        
        return {
            'image_shape': original_image.shape,
            'runway_regions': runway_regions,
            'total_runway_pixels': np.sum(binary_mask > 0),
            'runway_percentage': np.sum(binary_mask > 0) / binary_mask.size * 100
        }

def test_on_sample_images():
    """Test the model on sample images from the dataset"""
    detector = RunwayDetector()
    
    # Test on a few images from the dataset
    base_path = "../archive/1920x1080/1920x1080/split_70_10_20/test"
    test_images = [f for f in os.listdir(base_path) if f.endswith('.png')][:5]
    
    print(f"Testing on {len(test_images)} sample images...")
    
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(base_path, img_name)
        
        print(f"\n--- Testing Image {i+1}: {img_name} ---")
        
        # Detect runway region
        runway_info = detector.detect_runway_region(img_path)
        
        print(f"Image shape: {runway_info['image_shape']}")
        print(f"Total runway pixels: {runway_info['total_runway_pixels']}")
        print(f"Runway percentage: {runway_info['runway_percentage']:.2f}%")
        print(f"Number of runway regions detected: {len(runway_info['runway_regions'])}")
        
        if runway_info['runway_regions']:
            main_runway = runway_info['runway_regions'][0]
            print(f"Main runway region:")
            print(f"  Bounding box: {main_runway['bbox']}")
            print(f"  Center: {main_runway['center']}")
            print(f"  Area: {main_runway['area']}")
        
        # Create visualization
        detector.visualize_prediction(
            img_path, 
            threshold=0.5, 
            save_path=f'runway_detection_sample_{i+1}.png'
        )

def predict_single_image(image_path, threshold=0.5):
    """
    Simple function to predict runway for a single image
    
    Args:
        image_path: Path to the input image
        threshold: Threshold for binary mask (default: 0.5)
    """
    detector = RunwayDetector()
    return detector.detect_runway_region(image_path, threshold)

if __name__ == "__main__":
    # Test on sample images
    test_on_sample_images()
    
    # Example usage for single image prediction
    # image_path = "path_to_your_image.jpg"
    # result = predict_single_image(image_path)
    # print(result)

