import os
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class BasicRunwayDetector:
    """
    Basic runway detection using traditional computer vision
    Minimal dependencies - only numpy and PIL
    """
    
    def __init__(self):
        self.min_runway_area = 3000
        self.max_runway_area = 150000
        
    def load_image(self, image_path):
        """Load image using PIL"""
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def to_grayscale(self, image):
        """Convert to grayscale"""
        if len(image.shape) == 3:
            return np.mean(image, axis=2).astype(np.uint8)
        return image.astype(np.uint8)
    
    def simple_edge_detection(self, gray):
        """Simple edge detection using gradient"""
        h, w = gray.shape
        edges = np.zeros_like(gray)
        
        # Calculate gradients using Sobel-like operators
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Simple gradient calculation
                gx = abs(gray[i, j+1] - gray[i, j-1])
                gy = abs(gray[i+1, j] - gray[i-1, j])
                edges[i, j] = gx + gy  # Sum of absolute gradients
        
        # Normalize and threshold
        if edges.max() > 0:
            edges = (edges / edges.max() * 255).astype(np.uint8)
        
        # Apply threshold
        edges = np.where(edges > 25, 255, 0).astype(np.uint8)
        
        return edges
    
    def find_runway_regions(self, binary_image):
        """Find potential runway regions using simple connected component analysis"""
        h, w = binary_image.shape
        visited = np.zeros_like(binary_image, dtype=bool)
        regions = []
        
        def flood_fill(start_y, start_x):
            """Simple flood fill algorithm"""
            if (start_y < 0 or start_y >= h or start_x < 0 or start_x >= w or 
                visited[start_y, start_x] or binary_image[start_y, start_x] == 0):
                return []
            
            stack = [(start_y, start_x)]
            component = []
            
            while stack:
                y, x = stack.pop()
                if (y < 0 or y >= h or x < 0 or x >= w or 
                    visited[y, x] or binary_image[y, x] == 0):
                    continue
                
                visited[y, x] = True
                component.append((y, x))
                
                # Add 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        stack.append((y + dy, x + dx))
            
            return component
        
        # Find all connected components
        for i in range(h):
            for j in range(w):
                if binary_image[i, j] == 255 and not visited[i, j]:
                    component = flood_fill(i, j)
                    if len(component) > 500:  # Minimum size threshold
                        regions.append(component)
        
        return regions
    
    def analyze_region(self, region):
        """Analyze a region for runway characteristics"""
        if not region:
            return None
        
        ys, xs = zip(*region)
        ys, xs = np.array(ys), np.array(xs)
        
        area = len(region)
        height = ys.max() - ys.min()
        width = xs.max() - xs.min()
        
        if width == 0:
            aspect_ratio = float('inf')
        else:
            aspect_ratio = height / width
        
        center = (xs.mean(), ys.mean())
        bbox = (xs.min(), ys.min(), width, height)
        
        # Calculate compactness (how close to a rectangle)
        rect_area = height * width
        compactness = area / rect_area if rect_area > 0 else 0
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'center': center,
            'bbox': bbox,
            'region': region
        }
    
    def detect_runway_region(self, image_path):
        """
        Main function to detect runway region
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with runway detection results
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"Image shape: {image.shape}")
        
        # Convert to grayscale
        gray = self.to_grayscale(image)
        
        # Detect edges
        edges = self.simple_edge_detection(gray)
        
        # Find potential runway regions
        regions = self.find_runway_regions(edges)
        
        # Analyze regions
        runway_candidates = []
        for region in regions:
            analysis = self.analyze_region(region)
            if analysis:
                area = analysis['area']
                aspect_ratio = analysis['aspect_ratio']
                compactness = analysis['compactness']
                
                # Filter based on runway characteristics
                # Runways are typically long and narrow with good compactness
                if (self.min_runway_area < area < self.max_runway_area and 
                    compactness > 0.3 and  # Not too scattered
                    (aspect_ratio > 1.5 or aspect_ratio < 0.7)):  # Long and narrow
                    runway_candidates.append(analysis)
        
        # Sort by area (largest first)
        runway_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        results = {
            'image_shape': image.shape,
            'runway_candidates': runway_candidates,
            'total_candidates': len(runway_candidates),
            'main_runway': runway_candidates[0] if runway_candidates else None,
            'success': len(runway_candidates) > 0
        }
        
        # Print detailed results
        print(f"Found {len(runway_candidates)} runway candidates")
        
        if runway_candidates:
            main = runway_candidates[0]
            print(f"\nMain runway candidate:")
            print(f"  Area: {main['area']:,} pixels")
            print(f"  Aspect ratio: {main['aspect_ratio']:.2f}")
            print(f"  Compactness: {main['compactness']:.2f}")
            print(f"  Center: ({main['center'][0]:.1f}, {main['center'][1]:.1f})")
            print(f"  Bounding box: {main['bbox']}")
            
            # Calculate runway percentage
            total_pixels = image.shape[0] * image.shape[1]
            runway_percentage = (main['area'] / total_pixels) * 100
            print(f"  Runway coverage: {runway_percentage:.2f}%")
        else:
            print("No runway candidates found")
        
        return results
    
    def save_detection_mask(self, image_path, output_path=None):
        """Save the detection mask as an image"""
        results = self.detect_runway_region(image_path)
        
        if not results or not results['main_runway']:
            print("No runway detected to save")
            return None
        
        # Create mask image
        image = self.load_image(image_path)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        main_runway = results['main_runway']
        for y, x in main_runway['region']:
            mask[y, x] = 255
        
        # Save mask
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"runway_mask_{base_name}.png"
        
        mask_image = Image.fromarray(mask, mode='L')
        mask_image.save(output_path)
        print(f"Detection mask saved to: {output_path}")
        
        return output_path

def test_runway_detection():
    """Test the runway detection system"""
    detector = BasicRunwayDetector()
    
    # Test directory
    base_path = "../archive/1920x1080/1920x1080/split_70_10_20/test"
    
    if not os.path.exists(base_path):
        print(f"Test directory not found: {base_path}")
        print("Please ensure the dataset is properly organized")
        return []
    
    # Get test images
    test_images = [f for f in os.listdir(base_path) if f.endswith('.png')][:5]
    
    print(f"🚁 Basic Runway Detection System")
    print(f"Testing on {len(test_images)} images...")
    print("=" * 60)
    
    results = []
    successful_detections = 0
    
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(base_path, img_name)
        
        print(f"\n--- Image {i+1}/{len(test_images)}: {img_name} ---")
        
        result = detector.detect_runway_region(img_path)
        
        if result and result['success']:
            successful_detections += 1
            
            # Save detection mask
            mask_path = detector.save_detection_mask(img_path)
            
            results.append({
                'image': img_name,
                'success': True,
                'candidates': result['total_candidates'],
                'area': result['main_runway']['area'],
                'aspect_ratio': result['main_runway']['aspect_ratio'],
                'mask_saved': mask_path
            })
        else:
            results.append({
                'image': img_name,
                'success': False,
                'candidates': 0,
                'area': 0,
                'aspect_ratio': 0,
                'mask_saved': None
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(test_images)}")
    print(f"Successful detections: {successful_detections}")
    print(f"Success rate: {successful_detections/len(test_images)*100:.1f}%")
    
    if successful_detections > 0:
        avg_area = np.mean([r['area'] for r in results if r['success']])
        avg_aspect = np.mean([r['aspect_ratio'] for r in results if r['success']])
        print(f"Average runway area: {avg_area:,.0f} pixels")
        print(f"Average aspect ratio: {avg_aspect:.2f}")
    
    print(f"\n🎯 Detection Complete!")
    print("This traditional approach should give you a good baseline score.")
    
    return results

def predict_single_image(image_path):
    """Predict runway for a single image"""
    detector = BasicRunwayDetector()
    result = detector.detect_runway_region(image_path)
    
    if result and result['success']:
        # Save the detection mask
        mask_path = detector.save_detection_mask(image_path)
        return result, mask_path
    else:
        return result, None

if __name__ == "__main__":
    # Run the test
    results = test_runway_detection()
    
    print(f"\n💡 Tips for better results:")
    print("- Use higher resolution images if available")
    print("- Adjust threshold parameters for your specific dataset")
    print("- Consider deep learning approaches for higher accuracy")
    print("- This baseline should give you a decent score for the competition!")

