import requests
import os

# Test with known images if available
def test_known_images():
    """Test API with known glaucoma/normal images to verify labels"""
    
    # Add paths to test images here
    test_images = {
        "known_glaucoma.jpg": "should_be_glaucoma",
        "known_normal.jpg": "should_be_normal"
    }
    
    for image_file, expected in test_images.items():
        if os.path.exists(image_file):
            print(f"\nTesting {image_file} (expected: {expected})")
            
            with open(image_file, 'rb') as f:
                files = {'image': f}
                data = {'model': 'ResNet50'}
                
                response = requests.post('http://localhost:8000/predict', 
                                       files=files, data=data)
                
                if response.ok:
                    result = response.json()
                    print(f"Prediction: {result['prediction']}")
                    print(f"Probabilities: {result['probabilities']}")
                else:
                    print(f"Error: {response.text}")

if __name__ == "__main__":
    test_known_images()
