import os
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ImprovedRunwayDetector:
    """
    Improved runway detection with better parameter tuning
    Optimized for runway detection competition
    """
    
    def __init__(self):
        # Optimized parameters for better detection
        self.min_runway_area = 2000
        self.max_runway_area = 300000
        self.min_aspect_ratio = 0.1
        self.max_aspect_ratio = 10.0
        self.min_compactness = 0.2
        self.edge_threshold = 20  # Lowered for better edge detection
        
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
        """Convert to grayscale with better weighting"""
        if len(image.shape) == 3:
            # Use luminance formula for better grayscale conversion
            return (0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]).astype(np.uint8)
        return image.astype(np.uint8)
    
    def enhance_contrast(self, gray):
        """Enhance contrast for better edge detection"""
        # Apply histogram equalization-like enhancement
        hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Apply enhancement
        enhanced = np.interp(gray.flatten(), bins[:-1], cdf_normalized)
        return enhanced.reshape(gray.shape).astype(np.uint8)
    
    def advanced_edge_detection(self, gray):
        """Advanced edge detection with multiple thresholds"""
        h, w = gray.shape
        edges = np.zeros_like(gray)
        
        # Apply contrast enhancement
        enhanced = self.enhance_contrast(gray)
        
        # Calculate gradients using Sobel operators
        for i in range(1, h-1):
            for j in range(1, w-1):
                # Sobel X - convert to float to avoid overflow
                gx = (float(enhanced[i-1, j-1]) * -1 + float(enhanced[i-1, j+1]) * 1 +
                      float(enhanced[i, j-1]) * -2 + float(enhanced[i, j+1]) * 2 +
                      float(enhanced[i+1, j-1]) * -1 + float(enhanced[i+1, j+1]) * 1)
                
                # Sobel Y - convert to float to avoid overflow
                gy = (float(enhanced[i-1, j-1]) * -1 + float(enhanced[i-1, j]) * -2 + float(enhanced[i-1, j+1]) * -1 +
                      float(enhanced[i+1, j-1]) * 1 + float(enhanced[i+1, j]) * 2 + float(enhanced[i+1, j+1]) * 1)
                
                edges[i, j] = np.sqrt(gx*gx + gy*gy)
        
        # Normalize and apply threshold
        if edges.max() > 0:
            edges = (edges / edges.max() * 255).astype(np.uint8)
        
        # Apply adaptive threshold
        edges = np.where(edges > self.edge_threshold, 255, 0).astype(np.uint8)
        
        return edges
    
    def morphological_closing(self, binary_image):
        """Simple morphological closing to connect nearby edges"""
        h, w = binary_image.shape
        result = np.zeros_like(binary_image)
        
        # Simple dilation
        kernel_size = 3
        pad = kernel_size // 2
        padded = np.pad(binary_image, pad, mode='constant', constant_values=0)
        
        for i in range(h):
            for j in range(w):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                if np.any(region == 255):
                    result[i, j] = 255
        
        return result
    
    def find_runway_regions(self, binary_image):
        """Find potential runway regions with improved connectivity"""
        h, w = binary_image.shape
        visited = np.zeros_like(binary_image, dtype=bool)
        regions = []
        
        def flood_fill(start_y, start_x):
            """8-connected flood fill"""
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
                    if len(component) > 300:  # Lowered minimum size
                        regions.append(component)
        
        return regions
    
    def analyze_region(self, region):
        """Analyze a region for runway characteristics with improved metrics"""
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
        
        # Calculate compactness
        rect_area = height * width
        compactness = area / rect_area if rect_area > 0 else 0
        
        # Calculate density (how well the region fills its bounding box)
        density = area / rect_area if rect_area > 0 else 0
        
        # Calculate elongation (how stretched the region is)
        elongation = max(aspect_ratio, 1/aspect_ratio) if aspect_ratio > 0 else 0
        
        return {
            'area': area,
            'aspect_ratio': aspect_ratio,
            'compactness': compactness,
            'density': density,
            'elongation': elongation,
            'center': center,
            'bbox': bbox,
            'region': region
        }
    
    def score_runway_candidate(self, analysis):
        """Score a runway candidate based on multiple criteria"""
        if not analysis:
            return 0
        
        score = 0
        
        # Area score (prefer medium to large areas)
        area = analysis['area']
        if self.min_runway_area < area < self.max_runway_area:
            score += min(area / self.max_runway_area, 1.0) * 30
        
        # Aspect ratio score (prefer elongated shapes)
        aspect_ratio = analysis['aspect_ratio']
        if self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio:
            if aspect_ratio > 2 or aspect_ratio < 0.5:  # Elongated
                score += 25
        
        # Compactness score (prefer well-formed shapes)
        compactness = analysis['compactness']
        if compactness > self.min_compactness:
            score += compactness * 20
        
        # Density score
        density = analysis['density']
        score += density * 15
        
        # Elongation bonus
        elongation = analysis['elongation']
        if elongation > 3:  # Very elongated
            score += 10
        
        return score
    
    def detect_runway_region(self, image_path):
        """
        Main function to detect runway region with improved algorithm
        
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
        
        # Detect edges with enhancement
        edges = self.advanced_edge_detection(gray)
        
        # Apply morphological closing
        closed = self.morphological_closing(edges)
        
        # Find potential runway regions
        regions = self.find_runway_regions(closed)
        
        # Analyze and score regions
        scored_candidates = []
        for region in regions:
            analysis = self.analyze_region(region)
            if analysis:
                score = self.score_runway_candidate(analysis)
                if score > 30:  # Minimum score threshold
                    scored_candidates.append((score, analysis))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        runway_candidates = [analysis for score, analysis in scored_candidates]
        
        results = {
            'image_shape': image.shape,
            'runway_candidates': runway_candidates,
            'total_candidates': len(runway_candidates),
            'main_runway': runway_candidates[0] if runway_candidates else None,
            'success': len(runway_candidates) > 0,
            'scores': [score for score, analysis in scored_candidates]
        }
        
        # Print detailed results
        print(f"Found {len(runway_candidates)} high-scoring runway candidates")
        
        if runway_candidates:
            main = runway_candidates[0]
            main_score = scored_candidates[0][0] if scored_candidates else 0
            
            print(f"\nMain runway candidate (Score: {main_score:.1f}):")
            print(f"  Area: {main['area']:,} pixels")
            print(f"  Aspect ratio: {main['aspect_ratio']:.2f}")
            print(f"  Compactness: {main['compactness']:.2f}")
            print(f"  Density: {main['density']:.2f}")
            print(f"  Elongation: {main['elongation']:.2f}")
            print(f"  Center: ({main['center'][0]:.1f}, {main['center'][1]:.1f})")
            print(f"  Bounding box: {main['bbox']}")
            
            # Calculate runway percentage
            total_pixels = image.shape[0] * image.shape[1]
            runway_percentage = (main['area'] / total_pixels) * 100
            print(f"  Runway coverage: {runway_percentage:.2f}%")
        else:
            print("No high-scoring runway candidates found")
        
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
            output_path = f"improved_runway_mask_{base_name}.png"
        
        mask_image = Image.fromarray(mask, mode='L')
        mask_image.save(output_path)
        print(f"Detection mask saved to: {output_path}")
        
        return output_path

def test_improved_detection():
    """Test the improved runway detection system"""
    detector = ImprovedRunwayDetector()
    
    # Test directory
    base_path = "../archive/1920x1080/1920x1080/split_70_10_20/test"
    
    if not os.path.exists(base_path):
        print(f"Test directory not found: {base_path}")
        return []
    
    # Get test images (filter out hidden files)
    all_files = os.listdir(base_path)
    test_images = [f for f in all_files if f.endswith('.png') and not f.startswith('.')][:8]
    
    print(f"🚁 Improved Runway Detection System")
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
                'score': result['scores'][0] if result['scores'] else 0,
                'area': result['main_runway']['area'],
                'aspect_ratio': result['main_runway']['aspect_ratio'],
                'mask_saved': mask_path
            })
        else:
            results.append({
                'image': img_name,
                'success': False,
                'candidates': 0,
                'score': 0,
                'area': 0,
                'aspect_ratio': 0,
                'mask_saved': None
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("IMPROVED DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(test_images)}")
    print(f"Successful detections: {successful_detections}")
    print(f"Success rate: {successful_detections/len(test_images)*100:.1f}%")
    
    if successful_detections > 0:
        avg_score = np.mean([r['score'] for r in results if r['success']])
        avg_area = np.mean([r['area'] for r in results if r['success']])
        avg_aspect = np.mean([r['aspect_ratio'] for r in results if r['success']])
        print(f"Average detection score: {avg_score:.1f}")
        print(f"Average runway area: {avg_area:,.0f} pixels")
        print(f"Average aspect ratio: {avg_aspect:.2f}")
    
    print(f"\n🎯 Improved Detection Complete!")
    print("This enhanced approach should give you a much better score!")
    
    return results

def predict_single_image(image_path):
    """Predict runway for a single image using improved detector"""
    detector = ImprovedRunwayDetector()
    result = detector.detect_runway_region(image_path)
    
    if result and result['success']:
        # Save the detection mask
        mask_path = detector.save_detection_mask(image_path)
        return result, mask_path
    else:
        return result, None

if __name__ == "__main__":
    # Run the improved test
    results = test_improved_detection()
    
    print(f"\n🏆 Competition Tips:")
    print("- This improved detector should score well in the competition")
    print("- Adjust parameters based on your specific dataset")
    print("- Consider ensemble methods for even better results")
    print("- The scoring system helps identify the best candidates")
