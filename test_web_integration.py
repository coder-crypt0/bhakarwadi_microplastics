#!/usr/bin/env python3
"""
Test script for directory processing and web integration
Team Bhakarwadi - SIH 2025
"""

import os
import sys
import requests
import cv2
import numpy as np
import tempfile
import base64
from pathlib import Path

def test_directory_processing():
    """Test directory processing upload to web interface"""
    
    print("Testing directory processing integration...")
    
    # API configuration
    api_base = "https://thegroup11.com/sih/api"
    
    # Create a test image with some mock detection boxes
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Draw some mock detection boxes
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 2)
    cv2.rectangle(test_image, (300, 200), (400, 300), (0, 255, 0), 2)
    cv2.rectangle(test_image, (150, 350), (250, 420), (0, 255, 0), 2)
    
    # Add text to identify as test image
    cv2.putText(test_image, 'TEST DIRECTORY IMAGE', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, test_image)
        temp_path = tmp_file.name
    
    try:
        # Convert to base64
        with open(temp_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Prepare data
        data = {
            'image_data': image_data,
            'filename': 'test_directory_image.jpg',
            'count': 3,  # Mock count matching the boxes we drew
            'processing_time': 2.5,
            'source': 'directory_browse',
            'location': {
                'latitude': 19.0760,
                'longitude': 72.8777,
                'city': 'Mumbai',
                'country': 'India',
                'accuracy': 1000
            }
        }
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'MicroplasticDetector/1.0',
            'Accept': 'application/json'
        }
        
        # Upload to API
        print(f"Uploading test image to {api_base}/upload.php...")
        response = requests.post(f"{api_base}/upload.php", 
                               json=data, 
                               headers=headers,
                               timeout=30,
                               allow_redirects=False)
        
        # Handle redirect if needed
        if response.status_code in [301, 302]:
            redirect_url = response.headers.get('Location')
            print(f"Following redirect to: {redirect_url}")
            response = requests.post(redirect_url, 
                                   json=data, 
                                   headers=headers,
                                   timeout=30,
                                   allow_redirects=False)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: Test image uploaded successfully!")
            print(f"Response: {result}")
            
            # Test retrieval
            print("\nTesting retrieval...")
            get_response = requests.get(f"{api_base}/results.php?action=latest")
            if get_response.status_code == 200:
                get_data = get_response.json()
                print("Latest result retrieved:")
                print(f"Count: {get_data['data']['result']['microplastic_count']}")
                print(f"Source: {get_data['data']['result']['source']}")
                print(f"Image URL: {get_data['data']['result'].get('image_url', 'Not available')}")
            
            print("\nNow check the web interface at:")
            print("- Live Detection tab should show the directory image with LOCAL badge")
            print("- Browse Images tab should show the processed image with count")
            print("- Location Map tab should show the Mumbai marker")
            
        else:
            print(f"FAILED: Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_location_data():
    """Test location-based analytics"""
    print("\nTesting location analytics...")
    
    api_base = "https://thegroup11.com/sih/api"
    
    try:
        response = requests.get(f"{api_base}/results.php?action=history&limit=10")
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                results = data['data']['results']
                
                print(f"Found {len(results)} recent results")
                
                # Count results with location data
                with_location = [r for r in results if 'latitude' in r and 'longitude' in r]
                print(f"Results with location data: {len(with_location)}")
                
                if with_location:
                    print("Sample location data:")
                    sample = with_location[0]
                    print(f"  City: {sample.get('city', 'Unknown')}")
                    print(f"  Coordinates: {sample.get('latitude')}, {sample.get('longitude')}")
                    print(f"  Count: {sample.get('microplastic_count')}")
                
    except Exception as e:
        print(f"Location test error: {e}")

if __name__ == '__main__':
    test_directory_processing()
    test_location_data()
    
    print(f"\nTo view the results, open: https://thegroup11.com/sih/web/")
    print("Check all tabs to see the different features working!")