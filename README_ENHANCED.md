# 🔬 Microplastic Detection System - Team Bharatwadi

**Smart India Hackathon 2025** - Advanced Real-time Microplastic Detection with Modern Web Interface

## 🌟 Features

### 🎥 Real-time Detection
- Live camera feed processing with YOLOv8
- Automatic change detection (15-second intervals)
- Real-time upload to web interface
- Draggable overlay with detection count

### 🌐 Modern Web Interface
- **Dark theme** with modern UI/UX
- **Real-time updates** from camera feed
- **Image browser** with next/previous navigation
- **Analytics dashboard** with statistics
- **Responsive design** for all devices
- **Team Bharatwadi branding** throughout

### 📁 Directory Processing
- Browse and process entire image directories
- Batch processing with progress tracking
- Individual image navigation controls
- On-demand processing triggers

### 🔌 RESTful API
- PHP-based backend with JSON logging
- Upload/download endpoints for images
- Results and analytics endpoints
- Directory browsing API
- No SQL database required

## 🚀 Quick Setup

### 1. Prerequisites
```bash
# Python requirements
pip install ultralytics opencv-python PyQt5 requests numpy torch

# Web server (Apache/Nginx with PHP)
# Ensure PHP has curl and json extensions enabled
```

### 2. Project Structure
```
microplastic/
├── main_enhanced.py          # Enhanced PyQt5 application
├── best.pt                   # Your trained YOLOv8 model
├── sih/
│   ├── api/                  # PHP API endpoints
│   │   ├── config.php        # API configuration
│   │   ├── upload.php        # Image upload endpoint
│   │   ├── download.php      # Image download endpoint
│   │   ├── results.php       # Results and analytics
│   │   ├── browse.php        # Directory browsing
│   │   ├── index.php         # API documentation
│   │   ├── data/             # JSON logs storage
│   │   └── uploads/          # Uploaded images
│   └── web/                  # Modern web interface
│       ├── index.html        # Main webpage
│       ├── styles.css        # Dark theme styling
│       └── script.js         # JavaScript functionality
```

### 3. Web Server Setup

#### Option A: Local Development
```bash
# Using PHP built-in server
cd sih
php -S localhost:8000

# Access at: http://localhost:8000/web/
```

#### Option B: Production (Apache/Nginx)
1. Upload `sih/` folder to your web server
2. Ensure PHP has write permissions to `api/data/` and `api/uploads/`
3. Update `API_BASE_URL` in `main_enhanced.py` to your domain

### 4. Run the Application
```bash
# Basic usage
python main_enhanced.py

# With custom settings
python main_enhanced.py --camera 0 --api-url "http://thegroup11.com/sih/api"

# For directory processing
python main_enhanced.py --weights custom_model.pt
```

## 🔧 Configuration

### API URL Setup
Update the API URL in `main_enhanced.py`:
```python
API_BASE_URL = "http://thegroup11.com/sih/api"  # Your domain
```

### Camera Settings
- Default camera index: `0` (as specified)
- Change detection interval: `15 seconds`
- Auto-upload when no movement detected

### Detection Thresholds
```python
CONF_THRESHOLD = 0.03    # Confidence threshold
IOU_THRESHOLD = 0.30     # IoU threshold
MIN_CHANGE_THRESHOLD = 0.1  # Change detection sensitivity
```

## 📱 Web Interface Usage

### Live Detection Mode
1. **Automatic Updates**: Displays latest camera results every 15 seconds
2. **Manual Refresh**: Click "Refresh Now" for immediate update
3. **Pause/Resume**: Control auto-refresh functionality
4. **Real-time Stats**: View detection count and processing time

### Image Browser Mode
1. **Directory Path**: Enter path to image directory
2. **Load Directory**: Browse all images in the directory
3. **Navigation**: Use +/- buttons or arrow keys to navigate
4. **Processing**: Process individual images or entire directories

### Analytics Dashboard
1. **Statistics**: Total analyses, microplastic count, averages
2. **History**: Recent detection results with timestamps
3. **Performance**: Processing times and success rates

## 🎨 Web Interface Features

### Design Elements
- **Modern dark theme** with subtle gradients
- **Interactive animations** and hover effects
- **Responsive grid layouts** for all screen sizes
- **Professional color scheme**: Dark backgrounds with blue accents
- **Team Bharatwadi branding** with trophy and gradient badges

### Technical Features
- **Real-time API communication** with error handling
- **Progressive loading** with spinners and transitions
- **Responsive design** optimized for mobile and desktop
- **Accessibility features** with proper ARIA labels
- **Modern JavaScript** with class-based architecture

## 🔌 API Endpoints

### Upload Results
```http
POST /sih/api/upload.php
Content-Type: multipart/form-data

image: file
count: integer
processing_time: float (optional)
source: string (optional)
```

### Get Latest Result
```http
GET /sih/api/results.php?action=latest
```

### Get Analytics
```http
GET /sih/api/results.php?action=stats
```

### Browse Directory
```http
GET /sih/api/browse.php?action=list&directory=/path/to/images
```

### Download Image
```http
GET /sih/api/download.php?filename=image.jpg
```

## 🏆 Team Bharatwadi Integration

### Branding Elements
- **Logo**: Microscope icon with gradient text
- **Colors**: Professional blue (#2d6cdf) with accent gradients
- **Typography**: Modern Inter font family
- **Badges**: Trophy icons with gradient backgrounds
- **Watermarks**: Consistent branding throughout interface

### User Experience
- **Intuitive Navigation**: Tab-based interface with clear icons
- **Professional Appearance**: Dark theme with modern styling
- **Performance Optimized**: Fast loading and smooth animations
- **Mobile Responsive**: Works perfectly on all devices

## 🔧 Troubleshooting

### Common Issues

#### API Connection Failed
1. Verify web server is running
2. Check PHP extensions (curl, json)
3. Ensure proper file permissions
4. Update API_BASE_URL in main_enhanced.py

#### Camera Not Working
1. Check camera index (try 0, 1, 2)
2. Ensure camera is not used by other applications
3. Verify OpenCV installation

#### Model Loading Error
1. Ensure `best.pt` is in the correct path
2. Check PyTorch and Ultralytics installation
3. Verify model file is not corrupted

### Performance Tips
1. **GPU Acceleration**: Use CUDA if available
2. **Image Resolution**: Lower resolution for faster processing
3. **Confidence Threshold**: Adjust for optimal speed/accuracy balance
4. **Network Connection**: Ensure stable internet for API uploads

## 📄 License

This project is developed by **Team Bharatwadi** for **Smart India Hackathon 2025**.

## 🤝 Support

For technical support or questions:
- Visit: [thegroup11.com](http://thegroup11.com)
- Check API documentation at: `/sih/api/index.php`

---

**🏆 Team Bharatwadi** - *Advanced Microplastic Detection System*  
*Smart India Hackathon 2025*