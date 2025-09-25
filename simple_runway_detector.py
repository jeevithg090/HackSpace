import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class SimpleRunwayDetector:
    """
    Simple runway detection using traditional computer vision
    No external dependencies beyond numpy, matplotlib, and PIL
    """
    
    def __init__(self):
        self.min_runway_area = 5000
        self.max_runway_area = 200000
        
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
        
        # Calculate gradients
        for i in range(1, h-1):
            for j in range(1, w-1):
                gx = gray[i, j+1] - gray[i, j-1]
                gy = gray[i+1, j] - gray[i-1, j]
                edges[i, j] = np.sqrt(gx*gx + gy*gy)
        
        # Normalize and threshold
        edges = (edges / edges.max() * 255).astype(np.uint8)
        edges = np.where(edges > 30, 255, 0).astype(np.uint8)
        
        return edges
    
    def find_connected_components(self, binary_image):
        """Find connected components using simple flood fill"""
        h, w = binary_image.shape
        visited = np.zeros_like(binary_image, dtype=bool)
        components = []
        
        def flood_fill(start_y, start_x):
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
                
                # Add neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((y + dy, x + dx))
            
            return component
        
        for i in range(h):
            for j in range(w):
                if binary_image[i, j] == 255 and not visited[i, j]:
                    component = flood_fill(i, j)
                    if len(component) > 100:  # Minimum component size
                        components.append(component)
        
        return components
    
    def analyze_component(self, component):
        """Analyze a connected component for runway characteristics"""
        if not component:
            return None
        
        ys, xs = zip(*component)
        ys, xs = np.array(ys), np.array(xs)
        
        area = len(component)
        height = ys.max() - ys.min()
        width = xs.max() - xs.min()
        
        if width == 0:
            aspect_ratio = float('inf')
        else:
            aspect_ratio = height / width
        
        center = (xs.mean(), ys.mean())
        bbox = (xs.min(), ys.min(), width, height)
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'center': center,
            'bbox': bbox,
            'component': component
        }
    
    def detect_runway_region(self, image_path, visualize=True):
        """Main detection function"""
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
        
        # Find connected components
        components = self.find_connected_components(edges)
        
        # Analyze components
        runway_candidates = []
        for component in components:
            analysis = self.analyze_component(component)
            if analysis:
                area = analysis['area']
                aspect_ratio = analysis['aspect_ratio']
                
                # Filter by area and aspect ratio
                if (self.min_runway_area < area < self.max_runway_area and 
                    (aspect_ratio > 2 or aspect_ratio < 0.5)):
                    runway_candidates.append(analysis)
        
        # Sort by area
        runway_candidates.sort(key=lambda x: x['area'], reverse=True)
        
        results = {
            'image_shape': image.shape,
            'runway_candidates': runway_candidates,
            'total_candidates': len(runway_candidates),
            'main_runway': runway_candidates[0] if runway_candidates else None
        }
        
        # Print results
        print(f"Found {len(runway_candidates)} runway candidates")
        if runway_candidates:
            main = runway_candidates[0]
            print(f"Main runway:")
            print(f"  Area: {main['area']}")
            print(f"  Aspect ratio: {main['aspect_ratio']:.2f}")
            print(f"  Center: ({main['center'][0]:.1f}, {main['center'][1]:.1f})")
        
        # Visualize
        if visualize:
            self.visualize_detection(image, gray, edges, runway_candidates)
        
        return results
    
    def visualize_detection(self, original, gray, edges, runway_candidates):
        """Visualize the detection process"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grayscale
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        # Edges
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection')
        axes[1, 0].axis('off')
        
        # Final detection
        axes[1, 1].imshow(original)
        
        if runway_candidates:
            # Create mask for main runway
            main_runway = runway_candidates[0]
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            
            for y, x in main_runway['component']:
                mask[y, x] = 255
            
            # Overlay
            red_overlay = np.zeros((*mask.shape, 3))
            red_overlay[:, :, 0] = mask
            axes[1, 1].imshow(red_overlay, alpha=0.7)
            
            # Draw bounding box
            bbox = main_runway['bbox']
            from matplotlib.patches import Rectangle
            rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], 
                           fill=False, edgecolor='yellow', linewidth=3)
            axes[1, 1].add_patch(rect)
        
        axes[1, 1].set_title('Runway Detection')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

def test_runway_detection():
    """Test the detection system"""
    detector = SimpleRunwayDetector()
    
    # Test directory
    base_path = "../archive/1920x1080/1920x1080/split_70_10_20/test"
    
    if not os.path.exists(base_path):
        print(f"Test directory not found: {base_path}")
        return []
    
    # Get test images
    test_images = [f for f in os.listdir(base_path) if f.endswith('.png')][:3]
    
    print(f"Testing on {len(test_images)} images...")
    
    results = []
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(base_path, img_name)
        
        print(f"\n{'='*50}")
        print(f"Image {i+1}: {img_name}")
        print(f"{'='*50}")
        
        result = detector.detect_runway_region(img_path, visualize=True)
        
        if result:
            results.append({
                'image': img_name,
                'candidates': result['total_candidates'],
                'has_runway': result['main_runway'] is not None,
                'area': result['main_runway']['area'] if result['main_runway'] else 0
            })
    
    # Summary
    print(f"\n{'='*50}")
    print("DETECTION SUMMARY")
    print(f"{'='*50}")
    
    if results:
        total = len(results)
        detected = sum(1 for r in results if r['has_runway'])
        print(f"Images processed: {total}")
        print(f"Runways detected: {detected}")
        print(f"Success rate: {detected/total*100:.1f}%")
    
    return results

def predict_single_image(image_path):
    """Predict runway for a single image"""
    detector = SimpleRunwayDetector()
    return detector.detect_runway_region(image_path, visualize=True)

if __name__ == "__main__":
    print("🚁 Simple Runway Detection System")
    print("Using traditional computer vision techniques")
    print("=" * 50)
    
    # Test the system
    results = test_runway_detection()
    
    print(f"\n🎯 Detection Complete!")
    print("This approach should give you a good baseline score.")
    print("For better performance, consider using deep learning models.")

