# Directory Image Processing Guide - Team Bharatwadi

## 🔧 How to Use Directory Image Path

### Method 1: Through the GUI (Recommended)

1. **Launch the application:**
   ```bash
   python main_enhanced.py
   ```

2. **Switch to Directory Browser tab:**
   - Click on "📁 Directory Browser" tab in the interface

3. **Enter your image directory path:**
   ```
   Example paths:
   - Windows: C:\Users\ritik\Downloads\microplastic\microplastic\datasets\test_images\test_images
   - Windows (relative): .\datasets\test_images\test_images
   - Linux/Mac: /path/to/your/images
   ```

4. **Click "📁 Browse" button or press Enter**

5. **Navigate through images:**
   - Use **+** (next) and **-** (previous) buttons
   - Or use **arrow keys** on keyboard

6. **Process images:**
   - Click "⚡ Process Current Image" for single image
   - Click "⚡ Process All Images" for batch processing

### Method 2: Pre-configured Directory (Code Example)

Create a simple script to process a specific directory:

```python
import os
from pathlib import Path

# Set your image directory here
IMAGE_DIRECTORY = r"C:\Users\ritik\Downloads\microplastic\microplastic\datasets\test_images\test_images"

# Check if directory exists
if os.path.exists(IMAGE_DIRECTORY):
    print(f"✅ Directory found: {IMAGE_DIRECTORY}")
    
    # Count images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in Path(IMAGE_DIRECTORY).iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"📸 Found {len(image_files)} images")
    
    # Launch with directory pre-loaded (you can modify main_enhanced.py to accept this)
    os.system(f'python main_enhanced.py --directory "{IMAGE_DIRECTORY}"')
else:
    print(f"❌ Directory not found: {IMAGE_DIRECTORY}")
    print("Please check the path and try again.")
```

### Method 3: Command Line with Directory (Enhanced)

You can modify the main script to accept a directory parameter. Here's how to add it:

```python
# Add to argument parser in main_enhanced.py
parser.add_argument('--directory', help='Pre-load image directory for browsing')

# Then in the main window initialization:
if args.directory and os.path.exists(args.directory):
    # Auto-load the directory
    win.directory_input.setText(args.directory)
    win.load_directory()
```

## 📁 Directory Structure Examples

### Your Current Dataset Structure:
```
datasets/
├── test_images/
│   └── test_images/          ← Use this path!
│       ├── test_1.jpg
│       ├── test_10.jpg
│       ├── test_100.jpg
│       └── ... (more images)
├── train_images/
└── val_images/
```

### Supported Image Formats:
- ✅ JPEG (.jpg, .jpeg)
- ✅ PNG (.png)
- ✅ BMP (.bmp)  
- ✅ TIFF (.tiff)

### Example Valid Paths:

**Windows:**
```
C:\Users\ritik\Downloads\microplastic\microplastic\datasets\test_images\test_images
C:\Users\ritik\Downloads\microplastic\microplastic\datasets\train_images\train_images
.\datasets\test_images\test_images
```

**Linux/Mac:**
```
/home/user/microplastic/datasets/test_images/test_images
./datasets/test_images/test_images
~/microplastic/datasets/test_images/test_images
```

## 🚀 Quick Start Example

1. **Copy this path for your test images:**
   ```
   C:\Users\ritik\Downloads\microplastic\microplastic\datasets\test_images\test_images
   ```

2. **Run the application:**
   ```bash
   python main_enhanced.py
   ```

3. **In the GUI:**
   - Go to "📁 Directory Browser" tab
   - Paste the path in the text box
   - Click "📁 Browse" or press Enter
   - Use + and - buttons to navigate
   - Click "⚡ Process Current Image" to analyze

## 🔧 Troubleshooting

### Common Issues:

**"Directory not found" error:**
- ✅ Check if path exists using Windows File Explorer
- ✅ Use forward slashes `/` or double backslashes `\\` in paths
- ✅ Remove quotes from path if copy-pasted

**"No images found" error:**
- ✅ Ensure directory contains supported image formats
- ✅ Check if images are in subdirectories

**Permission denied:**
- ✅ Make sure you have read access to the directory
- ✅ Try running as administrator if needed

## 🎯 Processing Workflow

1. **Load Directory** → Scans for all supported images
2. **Navigate** → Browse through images one by one  
3. **Process** → Run YOLO detection on current image
4. **Upload** → Results automatically sent to web API
5. **View** → Check results on web interface

## 🌐 Web Integration

When you process directory images:
- Results are automatically logged to the web API
- View processed images at: `http://thegroup11.com/sih/web/`
- Switch to "Analytics" tab to see processing history
- All results include filename, count, and processing time

---

**🏆 Team Bharatwadi** - *Smart India Hackathon 2025*