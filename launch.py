#!/usr/bin/env python3
"""
Quick Launcher for Microplastic Detection System
Team Bharatwadi - SIH 2025

This script provides easy ways to launch the detection system with different configurations.
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    print("MICROPLASTIC DETECTION SYSTEM - Team Bharatwadi")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="Launch Microplastic Detection System")
    parser.add_argument('--mode', choices=['live', 'directory', 'gui'], default='gui',
                       help='Launch mode: live (camera only), directory (batch process), gui (full interface)')
    parser.add_argument('--directory', help='Image directory path for batch processing')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--api-url', default='http://thegroup11.com/sih/api', help='API base URL')
    parser.add_argument('--weights', default='best.pt', help='YOLO weights file')
    
    args = parser.parse_args()
    
    # Validate weights file
    if not os.path.exists(args.weights):
        print(f"ERROR: Weights file not found: {args.weights}")
        print("Please ensure 'best.pt' is in the current directory or specify correct path.")
        return 1
    
    # Build command
    cmd_parts = [
        sys.executable,
        'main_enhanced.py',
        f'--weights={args.weights}',
        f'--camera={args.camera}',
        f'--api-url={args.api_url}'
    ]
    
    if args.mode == 'live':
        print("STARTING LIVE CAMERA MODE...")
        print(f"Camera: {args.camera}")
        print(f"API: {args.api_url}")
        print("-" * 40)
        
    elif args.mode == 'directory':
        if not args.directory:
            print("DIRECTORY MODE - Please specify your image directory:")
            print()
            
            # Suggest common directories
            suggestions = [
                r".\datasets\test_images\test_images",
                r".\datasets\train_images\train_images", 
                r".\datasets\val_images\val_images"
            ]
            
            print("Common directories in your project:")
            for i, path in enumerate(suggestions, 1):
                exists = "[EXISTS]" if os.path.exists(path) else "[NOT FOUND]"
                print(f"  {i}. {exists} {path}")
            
            print()
            directory = input("Enter directory path (or number from above): ").strip()
            
            # Handle numbered selection
            if directory.isdigit() and 1 <= int(directory) <= len(suggestions):
                directory = suggestions[int(directory) - 1]
            
            if not directory:
                print("ERROR: No directory specified. Exiting.")
                return 1
                
            args.directory = directory
        
        # Validate directory
        if not os.path.exists(args.directory):
            print(f"ERROR: Directory not found: {args.directory}")
            return 1
            
        # Count images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in Path(args.directory).iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images in: {args.directory}")
        print(f"API: {args.api_url}")
        print("-" * 40)
        
    else:  # GUI mode
        print("Starting FULL GUI MODE...")
        print("Features available:")
        print("  - Live camera detection")
        print("  - Directory image browsing") 
        print("  - Analytics dashboard")
        print("  - Settings configuration")
        print(f"API: {args.api_url}")
        print("-" * 40)
    
    # Execute command
    cmd = ' '.join(f'"{part}"' if ' ' in part else part for part in cmd_parts)
    print(f"Executing: {cmd}")
    print()
    
    try:
        exit_code = os.system(cmd)
        return exit_code
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        return 0
    except Exception as e:
        print(f"ERROR: Error launching application: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)