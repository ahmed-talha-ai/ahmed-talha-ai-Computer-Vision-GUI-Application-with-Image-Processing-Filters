<!DOCTYPE html>
<html lang="en">
<head>
</head>
<body>
    <h1 align="center">Computer Vision GUI Application</h1>
    <p align="center">
        <strong>A Python-based image processing application with 17+ computer vision filters</strong>
    </p>

  <div class="badges" align="center">
        <span class="badge">
            <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python version">
        </span>
        <span class="badge">
            <img src="https://img.shields.io/badge/OpenCV-4.x-orange" alt="OpenCV">
        </span>
        <span class="badge">
            <img src="https://img.shields.io/badge/CustomTkinter-UI-yellowgreen" alt="CustomTkinter">
        </span>
    </div>

  <div align="center">
        <img src="screenshot.png" alt="Application Screenshot" width="600">
    </div>
    <h2>Features</h2>
    
  <div class="features">
        <div class="feature-list">
            <div class="feature-item">
                <strong>🖼️ Image Loading & Saving</strong><br>
                Supports common formats (PNG, JPG, JPEG, BMP)
            </div>
            <div class="feature-item">
                <strong>🔄 Real-time Processing</strong><br>
                Instant preview of filter effects
            </div>
            <div class="feature-item">
                <strong>🎚️ Parameter Control</strong><br>
                Adjustable sliders for fine-tuning filters
            </div>
            <div class="feature-item">
                <strong>🔄 Comparison View</strong><br>
                Side-by-side original/processed display
            </div>
            <div class="feature-item">
                <strong>🧹 One-click Reset</strong><br>
                Revert to original image
            </div>
            <div class="feature-item">
                <strong>🎨 Modern UI</strong><br>
                Dark mode and customizable theme
            </div>
        </div>
    </div>

   <h2>Available Filters</h2>
    
   <div class="filter-categories">
        <div class="filter-category">
            <h3>Noise Manipulation</h3>
            <div class="filter-list">
                <div class="filter-item">Add Noise</div>
                <div class="filter-item">Remove Noise</div>
                <div class="filter-item">Gaussian Noise</div>
            </div>
        </div>
             <div class="filter-category">
            <h3>Smoothing Filters</h3>
            <div class="filter-list">
                <div class="filter-item">Mean Filter</div>
                <div class="filter-item">Median Filter</div>
                <div class="filter-item">Gaussian Filter</div>
            </div>
        </div>
        
  <div class="filter-category">
            <h3>Morphological Operations</h3>
            <div class="filter-list">
                <div class="filter-item">Erosion</div>
                <div class="filter-item">Dilation</div>
                <div class="filter-item">Opening</div>
                <div class="filter-item">Closing</div>
            </div>
        </div>
        
 <div class="filter-category">
            <h3>Advanced Techniques</h3>
            <div class="filter-list">
                <div class="filter-item">Boundary Extraction</div>
                <div class="filter-item">Region Filling</div>
                <div class="filter-item">Global Threshold</div>
                <div class="filter-item">Adaptive Threshold</div>
                <div class="filter-item">Otsu Threshold</div>
                <div class="filter-item">Hough Transform</div>
                <div class="filter-item">Watershed Segmentation</div>
            </div>
        </div>
    </div>

  <h2>Technologies Used</h2>
    <ul>
        <li>Python 3.8+</li>
        <li>OpenCV (cv2) for image processing</li>
        <li>CustomTkinter for modern GUI</li>
        <li>NumPy for numerical operations</li>
        <li>PIL/Pillow for image handling</li>
    </ul>

   <h2>Installation</h2>
    <pre><code>pip install opencv-python customtkinter pillow numpy</code></pre>

  <h2>Usage</h2>
  <ol>
        <li>Run the application: <code>python Computer_vision_Project.py</code></li>
        <li>Click "Browse Image" to load an image</li>
        <li>Select filters from the sidebar</li>
        <li>Adjust parameters with sliders (when available)</li>
        <li>Save your processed image or reset to original</li>
    </ol>

  <h2>Contribution</h2>
    <p>Contributions are welcome! Please open an issue or pull request for any improvements or additional filters.</p>
    <h2>License</h2>
    <p>MIT License</p>
</body>
</html>
