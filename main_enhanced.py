"""Enhanced PyQt5 GUI for real-time microplastic detection with web API integration.

New Features:
- Automatic upload of results to web API
- Configurable change detection (15-second intervals)
- Directory browsing mode for batch processing
- API integration for seamless web interface communication
- Enhanced logging and error handling

Team Bhakarwadi - SIH 2025
"""

import sys
import os
import time
import cv2
import numpy as np
import torch
import requests
import json
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import pyqtSignal, Qt, QTimer
except Exception as e:
    print('PyQt5 not installed. Please install with: pip install PyQt5')
    raise

# Configuration
DEFAULT_WEIGHTS = os.path.join(os.path.dirname(__file__), 'best.pt')
DEFAULT_CAMERA_INDEX = 0  # Changed to 0 as specified
CONF_THRESHOLD = 0.03
IOU_THRESHOLD = 0.45
API_BASE_URL = "http://thegroup11.com/sih/api"  # Update this to your domain
CHANGE_DETECTION_INTERVAL = 15  # seconds
MIN_CHANGE_THRESHOLD = 0.1  # Minimum change ratio to trigger upload

class APIClient:
    """Client for communicating with the web API"""
    
    def __init__(self, base_url):
        self.base_url = base_url
    
    def upload_result(self, image_path, count, processing_time=None, source='camera', location=None):
        """Upload detection result to API"""
        try:
            # Get location if not provided and source is camera
            if location is None and source == 'camera':
                location = self.get_location_data()
            
            # Read image as base64 to avoid multipart issues
            import base64
            with open(image_path, 'rb') as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Send as JSON instead of multipart
            data = {
                'image_data': image_data,
                'filename': os.path.basename(image_path),
                'count': int(count),
                'processing_time': float(processing_time) if processing_time else None,
                'source': str(source)
            }
            
            # Add location data if available
            if location:
                data['location'] = location
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'MicroplasticDetector/1.0',
                'Accept': 'application/json'
            }
            
            # Try HTTPS first, then HTTP - disable redirects to prevent POST->GET conversion
            urls_to_try = [
                f"{self.base_url.replace('http://', 'https://')}/upload.php",
                f"{self.base_url}/upload.php"
            ]
            
            for url in urls_to_try:
                print(f"Trying URL: {url}")
                response = requests.post(url, 
                                       json=data, 
                                       headers=headers,
                                       timeout=30,
                                       allow_redirects=False)  # Prevent POST->GET redirect
                
                print(f"Response status: {response.status_code}")
                
                # Handle redirects manually
                if response.status_code in [301, 302, 303, 307, 308]:
                    redirect_url = response.headers.get('Location')
                    print(f"Redirect to: {redirect_url}")
                    if redirect_url:
                        # Make POST request to redirect URL
                        response = requests.post(redirect_url, 
                                               json=data, 
                                               headers=headers,
                                               timeout=30,
                                               allow_redirects=False)
                        print(f"Redirect response status: {response.status_code}")
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code != 405 and response.status_code not in [301, 302, 303, 307, 308]:
                    break  # Different error, don't retry with other URL
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Upload Error: {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return None
                
        except Exception as e:
            print(f"API Upload Exception: {e}")
            return None
    
    def log_directory_result(self, filename, count, processing_time=None):
        """Log directory processing result"""
        try:
            data = {
                'filename': str(filename),
                'count': int(count),
                'processing_time': float(processing_time) if processing_time else None
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'MicroplasticDetector/1.0',
                'Accept': 'application/json'
            }
            
            response = requests.post(
                f"{self.base_url}/browse.php",
                json=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Log Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"API Log Exception: {e}")
            return None
    
    def get_location_data(self):
        """Get location data using IP geolocation (fallback for desktop)"""
        try:
            # Try to get location using IP geolocation service
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    return {
                        'latitude': data.get('lat'),
                        'longitude': data.get('lon'),
                        'city': data.get('city'),
                        'country': data.get('country'),
                        'accuracy': 10000  # IP-based location is less accurate
                    }
        except Exception as e:
            print(f"Location detection failed: {e}")
        
        return None

class EnhancedInferenceThread(QtCore.QThread):
    """Enhanced inference thread with change detection and API integration"""
    frame_data = pyqtSignal(object)
    upload_completed = pyqtSignal(object)
    
    def __init__(self, weights, camera_index=0, device=None, api_client=None, parent=None):
        super().__init__(parent)
        self.weights = weights
        self.camera_index = camera_index
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.api_client = api_client
        self._run_flag = False
        self.conf = CONF_THRESHOLD
        self.iou = IOU_THRESHOLD
        self.model = None
        
        # Change detection variables
        self.last_frame = None
        self.last_upload_time = 0
        self.last_count = 0
        self.frame_buffer = []
        
    def run(self):
        try:
            self.model = YOLO(self.weights)
        except Exception as e:
            print('Failed to load model:', e)
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f'Cannot open camera index {self.camera_index}')
            return

        self._run_flag = True
        frame_count = 0
        
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1
            current_time = time.time()
            
            # Run inference
            try:
                start_time = time.time()
                results = self.model(frame, conf=self.conf, iou=self.iou, device=self.device)
                processing_time = time.time() - start_time
            except Exception as e:
                print('Inference error:', e)
                results = []
                processing_time = 0

            # Process results
            count = 0
            boxes_list = []
            if len(results) > 0:
                res = results[0]
                try:
                    xy = res.boxes.xyxy
                    if hasattr(xy, 'cpu'):
                        boxes = xy.cpu().numpy()
                    else:
                        boxes = np.array(xy)
                    if boxes.size != 0:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box[:4])
                            boxes_list.append([x1, y1, x2, y2])
                        count = len(boxes_list)
                except Exception:
                    count = 0

            # Convert to RGB and emit
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_data = {
                'frame': rgb_image.copy(),
                'boxes': boxes_list,
                'count': count,
                'processing_time': processing_time
            }
            self.frame_data.emit(frame_data)
            
            # Change detection and upload logic
            should_upload = False
            change_detected = False
            
            if self.last_frame is not None:
                # Calculate frame difference
                gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                
                diff = cv2.absdiff(gray_current, gray_last)
                change_ratio = np.sum(diff > 30) / (diff.shape[0] * diff.shape[1])
                
                if change_ratio > MIN_CHANGE_THRESHOLD:
                    change_detected = True
                    
            # Upload conditions:
            # 1. First result ever
            # 2. No change detected and 15 seconds passed
            # 3. Significant change detected and count changed
            time_since_upload = current_time - self.last_upload_time
            
            if (self.last_upload_time == 0 or 
                (not change_detected and time_since_upload >= CHANGE_DETECTION_INTERVAL) or
                (change_detected and count != self.last_count and time_since_upload >= 2)):
                should_upload = True
            
            if should_upload and self.api_client and count > 0:
                self.upload_frame_async(frame, count, processing_time)
                self.last_upload_time = current_time
                self.last_count = count
            
            self.last_frame = frame.copy()
            time.sleep(0.01)

        cap.release()
    
    def upload_frame_async(self, frame, count, processing_time):
        """Upload frame to API in a separate method"""
        try:
            # Save frame temporarily
            temp_path = f"temp_detection_{int(time.time())}.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Upload to API
            result = self.api_client.upload_result(temp_path, count, processing_time, 'camera')
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if result:
                self.upload_completed.emit({'status': 'success', 'result': result})
            else:
                self.upload_completed.emit({'status': 'error', 'message': 'Upload failed'})
                
        except Exception as e:
            print(f"Upload error: {e}")
            self.upload_completed.emit({'status': 'error', 'message': str(e)})

    def stop(self):
        self._run_flag = False
        self.wait()

class DirectoryProcessor(QtCore.QObject):
    """Process images from directory"""
    progress_updated = pyqtSignal(int, int)  # current, total
    result_ready = pyqtSignal(object)
    finished = pyqtSignal()
    
    def __init__(self, model, directory_path, api_client):
        super().__init__()
        self.model = model
        self.directory_path = directory_path
        self.api_client = api_client
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
    def process_directory(self):
        """Process all images in directory"""
        directory = Path(self.directory_path)
        if not directory.exists():
            return
            
        # Get all image files
        image_files = [f for f in directory.iterdir() 
                      if f.is_file() and f.suffix.lower() in self.image_extensions]
        
        total_files = len(image_files)
        
        for i, image_path in enumerate(image_files):
            try:
                # Load and process image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                    
                start_time = time.time()
                results = self.model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
                processing_time = time.time() - start_time
                
                # Count detections
                count = 0
                if len(results) > 0:
                    res = results[0]
                    if hasattr(res.boxes, 'xyxy'):
                        boxes = res.boxes.xyxy
                        if hasattr(boxes, 'cpu'):
                            boxes = boxes.cpu().numpy()
                        count = len(boxes) if boxes.size > 0 else 0
                
                # Log result to API
                if self.api_client:
                    self.api_client.log_directory_result(
                        image_path.name, count, processing_time
                    )
                
                # Emit result
                self.result_ready.emit({
                    'filename': image_path.name,
                    'count': count,
                    'processing_time': processing_time,
                    'path': str(image_path)
                })
                
                # Update progress
                self.progress_updated.emit(i + 1, total_files)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                
        self.finished.emit()

class EnhancedMainWindow(QtWidgets.QMainWindow):
    def __init__(self, weights=DEFAULT_WEIGHTS, camera_index=DEFAULT_CAMERA_INDEX, device=None, api_url=None):
        super().__init__()
        self.setWindowTitle('Microplastic Detector - Team Bhakarwadi (SIH 2025)')
        self.weights = weights
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.camera_index = camera_index  # Initialize before setup_ui
        
        # Initialize API client
        self.api_url = api_url or API_BASE_URL
        self.api_client = APIClient(self.api_url)
        
        self.setup_ui()
        self.setup_connections()
        
        # Initialize variables
        self.thread = None
        self.directory_processor = None
        self.directory_images = []  # List of image paths in directory
        
        # Status tracking
        self.total_detections = 0
        self.last_upload_status = None
        
    def setup_ui(self):
        """Setup the enhanced UI"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Header with branding
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Tab widget for different modes
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setObjectName('tabWidget')
        
        # Live detection tab
        live_tab = self.create_live_detection_tab()
        self.tab_widget.addTab(live_tab, "Live Detection")
        
        # Directory browser tab
        directory_tab = self.create_directory_tab()
        self.tab_widget.addTab(directory_tab, "Directory Browser")
        
        # Settings tab
        settings_tab = self.create_settings_tab()
        self.tab_widget.addTab(settings_tab, "Settings")
        
        main_layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        
        # Apply enhanced styles
        self.apply_enhanced_styles()
        
        # Initialize overlay
        self.overlay = DraggableLabel(parent=self)
        self.overlay.setFixedSize(200, 80)
        self.overlay.move(30, 30)
        self.overlay.show()
        
    def create_header(self):
        """Create header with Team Bhakarwadi branding"""
        header = QtWidgets.QFrame()
        header.setObjectName('header')
        layout = QtWidgets.QHBoxLayout(header)
        
        # Logo and title
        title_layout = QtWidgets.QHBoxLayout()
        title_icon = QtWidgets.QLabel("DETECT")
        title_icon.setStyleSheet("font-size: 18px; font-weight: bold; color: #2d6cdf;")
        title_text = QtWidgets.QLabel("Microplastic Detection System")
        title_text.setObjectName('titleText')
        
        title_layout.addWidget(title_icon)
        title_layout.addWidget(title_text)
        title_layout.addStretch()
        
        # Team badge
        team_badge = QtWidgets.QFrame()
        team_badge.setObjectName('teamBadge')
        team_layout = QtWidgets.QHBoxLayout(team_badge)
        team_layout.setContentsMargins(15, 8, 15, 8)
        
        trophy_icon = QtWidgets.QLabel("WINNER")
        trophy_icon.setStyleSheet("font-size: 12px; font-weight: bold;")
        team_text = QtWidgets.QLabel("Team Bhakarwadi")
        team_text.setObjectName('teamText')
        
        team_layout.addWidget(trophy_icon)
        team_layout.addWidget(team_text)
        
        layout.addLayout(title_layout)
        layout.addWidget(team_badge)
        
        return header
        
    def create_live_detection_tab(self):
        """Create live detection interface"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setSpacing(20)
        
        # Video display
        video_layout = QtWidgets.QVBoxLayout()
        self.video_widget = VideoWidget()
        self.video_widget.setMinimumSize(640, 480)
        video_layout.addWidget(self.video_widget)
        
        # Controls below video
        controls_layout = QtWidgets.QHBoxLayout()
        
        # Camera selection
        camera_label = QtWidgets.QLabel("Camera:")
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.camera_combo.setCurrentIndex(self.camera_index)
        
        controls_layout.addWidget(camera_label)
        controls_layout.addWidget(self.camera_combo)
        controls_layout.addSpacing(20)
        
        self.start_btn = QtWidgets.QPushButton('Start Detection')
        self.start_btn.setObjectName('primaryButton')
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.setEnabled(False)
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()
        
        video_layout.addLayout(controls_layout)
        layout.addLayout(video_layout, 2)
        
        # Stats panel
        stats_panel = self.create_stats_panel()
        layout.addWidget(stats_panel, 1)
        
        return widget
        
    def create_directory_tab(self):
        """Create directory processing interface with preview window and image list"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setSpacing(20)
        
        # Left side: Directory controls and image list
        left_panel = QtWidgets.QVBoxLayout()
        
        # Directory selection
        dir_layout = QtWidgets.QHBoxLayout()
        dir_layout.addWidget(QtWidgets.QLabel("Directory:"))
        
        self.directory_input = QtWidgets.QLineEdit()
        self.directory_input.setPlaceholderText("Enter path to image directory...")
        dir_layout.addWidget(self.directory_input)
        
        self.browse_btn = QtWidgets.QPushButton("Browse")
        dir_layout.addWidget(self.browse_btn)
        
        left_panel.addLayout(dir_layout)
        
        # Image list
        list_label = QtWidgets.QLabel("Images:")
        left_panel.addWidget(list_label)
        
        self.image_list = QtWidgets.QListWidget()
        self.image_list.setMinimumWidth(300)
        self.image_list.setMaximumWidth(400)
        left_panel.addWidget(self.image_list)
        
        # Directory controls
        dir_controls = QtWidgets.QHBoxLayout()
        self.load_dir_btn = QtWidgets.QPushButton("Load Directory")
        self.process_current_btn = QtWidgets.QPushButton("Process Selected")
        self.process_all_btn = QtWidgets.QPushButton("Process All")
        
        self.process_current_btn.setEnabled(False)
        self.process_all_btn.setEnabled(False)
        
        dir_controls.addWidget(self.load_dir_btn)
        dir_controls.addWidget(self.process_current_btn)
        dir_controls.addWidget(self.process_all_btn)
        
        left_panel.addLayout(dir_controls)
        
        # Progress bar
        self.dir_progress_bar = QtWidgets.QProgressBar()
        self.dir_progress_bar.setVisible(False)
        left_panel.addWidget(self.dir_progress_bar)
        
        # Add left panel to layout
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left_panel)
        layout.addWidget(left_widget, 1)
        
        # Right side: Preview window (similar to live detection)
        preview_layout = QtWidgets.QVBoxLayout()
        
        # Preview window
        self.directory_video_widget = VideoWidget()
        self.directory_video_widget.setMinimumSize(640, 480)
        preview_layout.addWidget(self.directory_video_widget)
        
        # Image info below preview
        info_layout = QtWidgets.QHBoxLayout()
        self.selected_image_label = QtWidgets.QLabel("No image selected")
        self.image_size_label = QtWidgets.QLabel("")
        self.detection_count_label = QtWidgets.QLabel("Count: --")
        
        info_layout.addWidget(self.selected_image_label)
        info_layout.addStretch()
        info_layout.addWidget(self.image_size_label)
        info_layout.addWidget(self.detection_count_label)
        
        preview_layout.addLayout(info_layout)
        
        # Add preview layout to main layout
        preview_widget = QtWidgets.QWidget()
        preview_widget.setLayout(preview_layout)
        layout.addWidget(preview_widget, 2)
        
        return widget
        
    def create_settings_tab(self):
        """Create settings interface"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # API Settings
        api_group = QtWidgets.QGroupBox("API Settings")
        api_layout = QtWidgets.QFormLayout(api_group)
        
        self.api_url_input = QtWidgets.QLineEdit(self.api_url)
        api_layout.addRow("API Base URL:", self.api_url_input)
        
        self.test_api_btn = QtWidgets.QPushButton("Test API Connection")
        api_layout.addRow(self.test_api_btn)
        
        layout.addWidget(api_group)
        
        # Detection Settings
        detection_group = QtWidgets.QGroupBox("Detection Settings")
        detection_layout = QtWidgets.QFormLayout(detection_group)
        
        self.conf_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 100)
        self.conf_slider.setValue(int(CONF_THRESHOLD * 100))
        self.conf_label = QtWidgets.QLabel(f"{CONF_THRESHOLD:.2f}")
        
        conf_layout = QtWidgets.QHBoxLayout()
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        detection_layout.addRow("Confidence Threshold:", conf_layout)
        
        layout.addWidget(detection_group)
        layout.addStretch()
        
        return widget
        
    def create_stats_panel(self):
        """Create statistics display panel"""
        panel = QtWidgets.QFrame()
        panel.setObjectName('statsPanel')
        layout = QtWidgets.QVBoxLayout(panel)
        
        # Main count display
        self.count_card = QtWidgets.QFrame()
        self.count_card.setObjectName('countCard')
        count_layout = QtWidgets.QVBoxLayout(self.count_card)
        
        self.count_label = QtWidgets.QLabel('0')
        self.count_label.setObjectName('countLabel')
        self.count_label.setAlignment(Qt.AlignCenter)
        
        count_subtitle = QtWidgets.QLabel('Microplastics Detected')
        count_subtitle.setObjectName('countSubtitle')
        count_subtitle.setAlignment(Qt.AlignCenter)
        
        count_layout.addWidget(self.count_label)
        count_layout.addWidget(count_subtitle)
        layout.addWidget(self.count_card)
        
        # Additional stats
        self.stats_list = QtWidgets.QListWidget()
        self.stats_list.setObjectName('statsList')
        self.stats_list.setMaximumHeight(200)
        layout.addWidget(self.stats_list)
        
        # Upload status
        self.upload_status = QtWidgets.QLabel("Upload Status: Ready")
        self.upload_status.setObjectName('uploadStatus')
        layout.addWidget(self.upload_status)
        
        layout.addStretch()
        return panel
        
    def setup_connections(self):
        """Setup signal connections"""
        self.start_btn.clicked.connect(self.start_inference)
        self.stop_btn.clicked.connect(self.stop_inference)
        self.browse_btn.clicked.connect(self.browse_directory)
        self.test_api_btn.clicked.connect(self.test_api_connection)
        
        # New directory tab connections
        self.load_dir_btn.clicked.connect(self.load_directory)
        self.process_current_btn.clicked.connect(self.process_current_image)
        self.process_all_btn.clicked.connect(self.process_all_images)
        self.image_list.itemClicked.connect(self.on_image_selected)
        
        # Camera selection connection
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        
        # Slider connections
        self.conf_slider.valueChanged.connect(self.update_confidence_threshold)
        
    def start_inference(self):
        """Start live detection with API integration"""
        if not os.path.exists(self.weights):
            QtWidgets.QMessageBox.critical(self, 'Model Error', f'Weights file not found:\n{self.weights}')
            return

        # Get selected camera index
        selected_camera = self.camera_combo.currentIndex()
        
        self.thread = EnhancedInferenceThread(
            self.weights,
            camera_index=selected_camera,
            device=self.device,
            api_client=self.api_client
        )
        
        self.thread.frame_data.connect(self.on_frame_data)
        self.thread.upload_completed.connect(self.on_upload_completed)
        self.thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.showMessage('🔴 Live detection started - Auto-uploading to web interface')

    def stop_inference(self):
        """Stop live detection"""
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.showMessage('⏹️ Detection stopped')

    def on_frame_data(self, data):
        """Handle incoming frame data"""
        frame = data['frame']
        boxes = data['boxes']
        count = data['count']
        processing_time = data.get('processing_time', 0)
        
        # Update video display
        self.video_widget.set_frame_and_boxes(frame, boxes)
        
        # Update overlay and main count
        self.overlay.setText(f'Microplastic\n{count}')
        self.count_label.setText(str(count))
        
        # Update stats
        self.total_detections += count
        self.update_stats_display(count, processing_time)

    def on_upload_completed(self, result):
        """Handle API upload completion"""
        if result['status'] == 'success':
            self.upload_status.setText("Upload: Success")
            self.upload_status.setStyleSheet("color: #4CAF50;")
        else:
            self.upload_status.setText(f"Upload: {result.get('message', 'Failed')}")
            self.upload_status.setStyleSheet("color: #f44336;")

    def update_stats_display(self, current_count, processing_time):
        """Update statistics display"""
        stats_items = [
            f"Current Detection: {current_count}",
            f"Total Found: {self.total_detections}",
            f"Processing: {processing_time:.3f}s",
            f"API: {self.api_url}",
        ]
        
        self.stats_list.clear()
        for item in stats_items:
            self.stats_list.addItem(item)

    def browse_directory(self):
        """Browse for image directory"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Image Directory"
        )
        if directory:
            self.directory_input.setText(directory)

    def on_camera_changed(self, index):
        """Handle camera selection change"""
        self.camera_index = index
        # If currently running, restart with new camera
        if hasattr(self, 'thread') and self.thread is not None:
            self.stop_inference()
            self.start_inference()

    def load_directory(self):
        """Load images from directory into the list"""
        directory_path = self.directory_input.text().strip()
        if not directory_path or not os.path.exists(directory_path):
            QtWidgets.QMessageBox.warning(self, "Invalid Directory", "Please select a valid directory")
            return
        
        # Clear existing items
        self.image_list.clear()
        self.directory_images = []
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        try:
            for filename in sorted(os.listdir(directory_path)):
                filepath = os.path.join(directory_path, filename)
                if os.path.isfile(filepath):
                    _, ext = os.path.splitext(filename.lower())
                    if ext in image_extensions:
                        self.directory_images.append(filepath)
                        self.image_list.addItem(filename)
            
            if self.directory_images:
                self.process_current_btn.setEnabled(True)
                self.process_all_btn.setEnabled(True)
                self.selected_image_label.setText(f"Loaded {len(self.directory_images)} images")
                
                # Select first image
                self.image_list.setCurrentRow(0)
                self.on_image_selected(self.image_list.item(0))
            else:
                QtWidgets.QMessageBox.information(self, "No Images", "No supported images found in directory")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load directory: {e}")

    def on_image_selected(self, item):
        """Handle image selection from list"""
        if not item:
            return
            
        current_row = self.image_list.row(item)
        if current_row < 0 or current_row >= len(self.directory_images):
            return
            
        image_path = self.directory_images[current_row]
        
        try:
            # Load and display image
            image = cv2.imread(image_path)
            if image is not None:
                # Convert to RGB for Qt
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Display in preview window (no boxes initially)
                self.directory_video_widget.set_frame_and_boxes(rgb_image, [])
                
                # Update info labels
                filename = os.path.basename(image_path)
                file_size = os.path.getsize(image_path)
                size_str = f"{file_size // 1024} KB" if file_size > 1024 else f"{file_size} B"
                
                self.selected_image_label.setText(filename)
                self.image_size_label.setText(f"{image.shape[1]}x{image.shape[0]} - {size_str}")
                self.detection_count_label.setText("Count: --")
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image: {e}")

    def process_current_image(self):
        """Process currently selected image"""
        current_row = self.image_list.currentRow()
        if current_row < 0 or current_row >= len(self.directory_images):
            return
            
        image_path = self.directory_images[current_row]
        self._process_single_image(image_path)

    def process_all_images(self):
        """Process all images in the directory"""
        if not self.directory_images:
            return
            
        # Show progress
        self.dir_progress_bar.setMaximum(len(self.directory_images))
        self.dir_progress_bar.setValue(0)
        self.dir_progress_bar.setVisible(True)
        
        # Disable buttons
        self.process_current_btn.setEnabled(False)
        self.process_all_btn.setEnabled(False)
        
        # Process all images
        for i, image_path in enumerate(self.directory_images):
            self._process_single_image(image_path)
            self.dir_progress_bar.setValue(i + 1)
            QtWidgets.QApplication.processEvents()  # Keep UI responsive
        
        # Hide progress and re-enable buttons
        self.dir_progress_bar.setVisible(False)
        self.process_current_btn.setEnabled(True)
        self.process_all_btn.setEnabled(True)

    def _process_single_image(self, image_path):
        """Process a single image and update display"""
        try:
            # Load model if needed
            if not hasattr(self, 'directory_model'):
                self.directory_model = YOLO(self.weights)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return
            
            # Run detection
            start_time = time.time()
            results = self.directory_model(image, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)
            processing_time = time.time() - start_time
            
            # Process results
            count = 0
            boxes_list = []
            if len(results) > 0:
                res = results[0]
                try:
                    xy = res.boxes.xyxy
                    if hasattr(xy, 'cpu'):
                        boxes = xy.cpu().numpy()
                    else:
                        boxes = np.array(xy)
                    if boxes.size != 0:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box[:4])
                            boxes_list.append([x1, y1, x2, y2])
                        count = len(boxes_list)
                except Exception:
                    count = 0
            
            # Update display if this is the selected image
            current_row = self.image_list.currentRow()
            if current_row >= 0 and current_row < len(self.directory_images):
                if self.directory_images[current_row] == image_path:
                    # Convert to RGB and display with boxes
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.directory_video_widget.set_frame_and_boxes(rgb_image, boxes_list)
                    self.detection_count_label.setText(f"Count: {count}")
            
            # Log to API and upload processed image for web interface
            if self.api_client:
                filename = os.path.basename(image_path)
                
                # Create a temporary processed image with bounding boxes for web display
                if boxes_list:
                    display_image = image.copy()
                    for box in boxes_list:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Save processed image temporarily
                    temp_processed = f"temp_processed_{os.path.basename(image_path)}"
                    cv2.imwrite(temp_processed, display_image)
                    
                    # Upload to API
                    self.api_client.upload_result(temp_processed, count, processing_time, 'directory_browse')
                    
                    # Clean up temp file
                    if os.path.exists(temp_processed):
                        os.remove(temp_processed)
                else:
                    # No detections, just log the result
                    self.api_client.log_directory_result(filename, count, processing_time)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    def test_api_connection(self):
        """Test API connection"""
        try:
            response = requests.get(f"{self.api_url_input.text().strip()}/index.php", timeout=5)
            if response.status_code == 200:
                QtWidgets.QMessageBox.information(self, "API Test", "✅ API connection successful!")
            else:
                QtWidgets.QMessageBox.warning(self, "API Test", f"❌ API returned status: {response.status_code}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "API Test", f"❌ Connection failed: {e}")

    def update_confidence_threshold(self, value):
        """Update confidence threshold"""
        threshold = value / 100.0
        self.conf_label.setText(f"{threshold:.2f}")
        if self.thread:
            self.thread.conf = threshold

    def apply_enhanced_styles(self):
        """Apply enhanced styling with Team Bharatwadi theme"""
        stylesheet = """
        QMainWindow { 
            background: #0b0b0b; 
            color: #eaeaea;
        }
        
        #header {
            background: linear-gradient(135deg, #1f1f1f, #2a2a2a);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 15px;
            margin-bottom: 10px;
        }
        
        #titleText {
            font-size: 24px;
            font-weight: bold;
            color: #2d6cdf;
        }
        
        #teamBadge {
            background: linear-gradient(135deg, #ff6b6b, #feca57);
            border-radius: 20px;
            color: #0b0b0b;
        }
        
        #teamText {
            font-weight: bold;
            color: #0b0b0b;
        }
        
        QTabWidget::pane {
            border: 1px solid #333;
            background: #1f1f1f;
            border-radius: 8px;
        }
        
        QTabBar::tab {
            background: #2a2a2a;
            color: #bbb;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        
        QTabBar::tab:selected {
            background: #2d6cdf;
            color: white;
        }
        
        #countCard {
            background: linear-gradient(135deg, #2d6cdf, #4CAF50);
            border-radius: 15px;
            padding: 20px;
            margin: 10px;
        }
        
        #countLabel {
            font-size: 48px;
            font-weight: bold;
            color: white;
        }
        
        #countSubtitle {
            font-size: 14px;
            color: rgba(255,255,255,0.9);
        }
        
        #statsPanel {
            background: #1f1f1f;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 15px;
        }
        
        #statsList {
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 8px;
            color: #eaeaea;
        }
        
        #uploadStatus {
            font-weight: bold;
            padding: 8px;
            border-radius: 6px;
            background: #2a2a2a;
        }
        
        QPushButton {
            background: #2a2a2a;
            color: #eaeaea;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: 500;
        }
        
        QPushButton:hover {
            background: #333;
            border-color: rgba(255,255,255,0.2);
        }
        
        #primaryButton {
            background: #2d6cdf;
            color: white;
            border-color: #2d6cdf;
        }
        
        #primaryButton:hover {
            background: #1e4bb8;
        }
        
        QPushButton:disabled {
            background: #1a1a1a;
            color: #666;
        }
        
        QLineEdit {
            background: #2a2a2a;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px;
            padding: 8px 12px;
            color: #eaeaea;
            font-size: 14px;
        }
        
        QLineEdit:focus {
            border-color: #2d6cdf;
        }
        
        QGroupBox {
            font-weight: bold;
            color: #2d6cdf;
            border: 2px solid #333;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 8px;
        }
        
        QProgressBar {
            border: 1px solid #333;
            border-radius: 6px;
            background: #2a2a2a;
            text-align: center;
            color: white;
        }
        
        QProgressBar::chunk {
            background: linear-gradient(to right, #2d6cdf, #4CAF50);
            border-radius: 5px;
        }
        
        QTextEdit {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #eaeaea;
            font-family: 'Consolas', monospace;
            font-size: 12px;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #333;
            height: 6px;
            background: #2a2a2a;
            border-radius: 3px;
        }
        
        QSlider::handle:horizontal {
            background: #2d6cdf;
            border: 1px solid #2d6cdf;
            width: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        
        QSlider::sub-page:horizontal {
            background: #2d6cdf;
            border-radius: 3px;
        }
        """
        self.setStyleSheet(stylesheet)

    def closeEvent(self, event):
        """Handle window close"""
        if self.thread is not None:
            self.thread.stop()
        event.accept()

# Keep the existing VideoWidget and DraggableLabel classes
class VideoWidget(QtWidgets.QWidget):
    """Widget that displays frames, supports pan & zoom and draws boxes scaled to view."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pix = None
        self._boxes = []
        self._scale = 1.0
        self._offset = QtCore.QPointF(0, 0)
        self._dragging = False
        self.setMouseTracking(True)

    def set_frame_and_boxes(self, rgb_frame, boxes):
        h, w = rgb_frame.shape[:2]
        image = QtGui.QImage(rgb_frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self._pix = QtGui.QPixmap.fromImage(image)
        self._boxes = boxes
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#0b0b0b'))
        if self._pix is None:
            return

        sw = self._pix.width() * self._scale
        sh = self._pix.height() * self._scale
        cx = (self.width() - sw) / 2 + self._offset.x()
        cy = (self.height() - sh) / 2 + self._offset.y()

        tx = int(cx)
        ty = int(cy)
        tw = max(1, int(sw))
        th = max(1, int(sh))
        painter.drawPixmap(tx, ty, tw, th, self._pix)

        pen = QtGui.QPen(QtGui.QColor(0, 255, 128), max(2, int(2 * self._scale)))
        painter.setPen(pen)
        for b in self._boxes:
            x1, y1, x2, y2 = b
            rx1 = cx + x1 * self._scale
            ry1 = cy + y1 * self._scale
            rx2 = cx + x2 * self._scale
            ry2 = cy + y2 * self._scale
            painter.drawRoundedRect(QtCore.QRectF(rx1, ry1, rx2 - rx1, ry2 - ry1), 4, 4)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.0 + (0.001 * delta)
        self._scale = max(0.1, min(5.0, self._scale * factor))
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if getattr(self, '_dragging', False):
            delta = event.pos() - self._last_pos
            self._offset += delta
            self._last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self._dragging = False

class DraggableLabel(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.label = QtWidgets.QLabel('Microplastic\n0', self)
        self.label.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont('Helvetica Neue', 14)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setStyleSheet('color: white;')
        self.setStyleSheet('background: rgba(20,20,20,0.85); border-radius: 10px;')
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label)
        self._drag_active = False

    def setText(self, text):
        self.label.setText(text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_active = True
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self._drag_active:
            self.move(event.globalPos() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_active = False

def main():
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default=DEFAULT_WEIGHTS)
    parser.add_argument('--camera', default=DEFAULT_CAMERA_INDEX, type=int)
    parser.add_argument('--device', default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--api-url', default=API_BASE_URL, help='Base URL for the web API')
    args = parser.parse_args()

    # Update API URL if provided - use provided URL directly
    api_url = args.api_url

    win = EnhancedMainWindow(weights=args.weights, camera_index=args.camera, device=args.device, api_url=api_url)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()