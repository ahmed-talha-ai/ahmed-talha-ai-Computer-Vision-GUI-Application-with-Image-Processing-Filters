#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🧠 Computer Vision GUI Project")
        self.geometry("1400x800")
        self.image = None
        self.original_image = None
        self.processed_image = None
        self.active_filters = {}  # Track active filter buttons
        self.slider_frame = None  # Frame for sliders
        self.sliders = {}  # Track sliders

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=250)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nswe", padx=10, pady=10)

        # Title
        self.title_label = ctk.CTkLabel(self, text="🧠 Computer Vision GUI Project", font=ctk.CTkFont(size=28, weight="bold"))
        self.title_label.grid(row=0, column=1, columnspan=2, pady=(15, 0), sticky="n")

        # Image labels
        self.original_label = ctk.CTkLabel(self, text="Original Image", font=ctk.CTkFont(size=16, weight="bold"))
        self.original_label.grid(row=1, column=1, padx=(10, 5), pady=(0, 5), sticky="n")
        
        self.processed_label = ctk.CTkLabel(self, text="Processed Image", font=ctk.CTkFont(size=16, weight="bold"))
        self.processed_label.grid(row=1, column=2, padx=(5, 10), pady=(0, 5), sticky="n")

        # Important Buttons
        self.browse_button = ctk.CTkButton(self.sidebar, text="📁 Browse Image", command=self.load_image, fg_color="royalblue")
        self.browse_button.pack(pady=(20, 10), fill="x")

        self.reset_button = ctk.CTkButton(self.sidebar, text="🔁 Reset Image", command=self.reset_image, fg_color="orangered")
        self.reset_button.pack(pady=5, fill="x")

        self.save_button = ctk.CTkButton(self.sidebar, text="💾 Save Image", command=self.save_image, fg_color="seagreen")
        self.save_button.pack(pady=5, fill="x")

        # Filter Buttons
        self.filters = {
            "Add Noise": (self.add_noise, {"std": (10, 50, 25)}),
            "Remove Noise": (self.remove_noise, None),
            "Mean Filter": (self.mean_filter, {"kernel_size": (3, 9, 3, 2)}),
            "Median Filter": (self.median_filter, {"kernel_size": (3, 9, 3, 2)}),
            "Gaussian Filter": (self.gaussian_filter, {"kernel_size": (3, 9, 3, 2)}),
            "Gaussian Noise": (self.gaussian_noise, {"std": (10, 50, 25)}),
            "Erosion": (self.erosion, {"kernel_size": (3, 7, 3, 2)}),
            "Dilation": (self.dilation, {"kernel_size": (3, 7, 3, 2)}),
            "Opening": (self.opening, {"kernel_size": (3, 7, 3, 2)}),
            "Closing": (self.closing, {"kernel_size": (3, 7, 3, 2)}),
            "Boundary Extraction": (self.boundary_extraction, {"kernel_size": (3, 7, 3, 2)}),
            "Region Filling": (self.region_filling, None),
            "Global Threshold": (self.global_threshold, {"thresh": (50, 200, 127)}),
            "Adaptive Threshold": (self.adaptive_threshold, None),
            "Otsu Threshold": (self.otsu_threshold, None),
            "Hough": (self.hough_transform, {"threshold": (50, 150, 100)}),
            "Watershed": (self.watershed_segmentation, None),
        }

        self.filter_buttons = {}
        for name, (func, params) in self.filters.items():
            btn = ctk.CTkButton(self.sidebar, text=name, command=lambda n=name, f=func, p=params: self.apply_filter(n, f, p), fg_color="#4682B4")
            btn.pack(pady=3, fill="x")
            self.filter_buttons[name] = btn

        # Original Image Display
        self.original_image_label = ctk.CTkLabel(self, text="No image loaded", anchor="center")
        self.original_image_label.grid(row=1, column=1, padx=(10, 5), pady=(30, 10), sticky="nsew")

        # Processed Image Display
        self.processed_image_label = ctk.CTkLabel(self, text="No image loaded", anchor="center")
        self.processed_image_label.grid(row=1, column=2, padx=(5, 10), pady=(30, 10), sticky="nsew")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image = cv2.imread(file_path)
            self.original_image = self.image.copy()
            self.processed_image = self.image.copy()
            self.display_images()
            self.clear_sliders()
            # Reset all filter button colors to steel blue
            for name, btn in self.filter_buttons.items():
                btn.configure(fg_color="#4682B4")
            self.active_filters.clear()

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.display_images()
            self.clear_sliders()
            # Reset all filter button colors to steel blue
            for name, btn in self.filter_buttons.items():
                btn.configure(fg_color="#4682B4")
            self.active_filters.clear()

    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
            if file_path:
                cv2.imwrite(file_path, self.processed_image)

    def display_images(self):
        if self.original_image is not None:
            # Display original image
            self.display_single_image(self.original_image, self.original_image_label)
            
            # Display processed image
            self.display_single_image(self.processed_image, self.processed_image_label)

    def display_single_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        # Adjust size for side-by-side display
        img_pil = img_pil.resize((500, 400), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        label.configure(image=imgtk, text="")
        label.image = imgtk

    def update_and_display(self, img):
        self.processed_image = img
        self.display_images()

    def clear_sliders(self):
        if self.slider_frame:
            self.slider_frame.destroy()
            self.sliders.clear()
        self.slider_frame = ctk.CTkFrame(self.sidebar)
        self.slider_frame.pack(pady=10, fill="x")

    def create_slider(self, label, min_val, max_val, default_val, step=1):
        ctk.CTkLabel(self.slider_frame, text=label).pack()
        slider = ctk.CTkSlider(self.slider_frame, from_=min_val, to=max_val, number_of_steps=int((max_val - min_val) / step))
        slider.set(default_val)
        slider.pack(pady=5, fill="x")
        return slider

    def apply_filter(self, name, func, params):
        if self.processed_image is None:
            return
        # Change the clicked button's color to green
        self.filter_buttons[name].configure(fg_color="#4CAF50")
        self.active_filters[name] = True
        # Reset other filter buttons to steel blue
        for btn_name, btn in self.filter_buttons.items():
            if btn_name != name and btn_name not in self.active_filters:
                btn.configure(fg_color="#4682B4")
        # Clear previous sliders
        self.clear_sliders()
        # Create sliders for parameters
        if params:
            for param_name, (min_val, max_val, default_val, *step) in params.items():
                step = step[0] if step else 1
                self.sliders[param_name] = self.create_slider(param_name, min_val, max_val, default_val, step)
        # Apply filter with default or slider values
        self.apply_filter_with_params(func, params)

    def apply_filter_with_params(self, func, params):
        if params:
            param_values = {name: slider.get() for name, slider in self.sliders.items()}
            for name, value in param_values.items():
                if name in ["kernel_size"]:  # Ensure kernel size is odd
                    param_values[name] = int(value) | 1
                else:
                    param_values[name] = int(value) if value.is_integer() else value
            func(**param_values)
        else:
            func()

    # === Filters ===
    def add_noise(self, std=25):
        noise = np.random.normal(0, std, self.processed_image.shape).astype(np.uint8)
        self.update_and_display(cv2.add(self.processed_image, noise))

    def remove_noise(self):
        denoised = cv2.fastNlMeansDenoisingColored(self.processed_image, None, 15, 15, 7, 21)
        self.update_and_display(denoised)

    def mean_filter(self, kernel_size=3):
        kernel_size = int(kernel_size) | 1  # Ensure odd
        self.update_and_display(cv2.blur(self.processed_image, (kernel_size, kernel_size)))

    def median_filter(self, kernel_size=3):
        kernel_size = int(kernel_size) | 1  # Ensure odd
        self.update_and_display(cv2.medianBlur(self.processed_image, kernel_size))

    def gaussian_filter(self, kernel_size=3):
        kernel_size = int(kernel_size) | 1  # Ensure odd
        self.update_and_display(cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), 0))

    def gaussian_noise(self, std=25):
        row, col, ch = self.processed_image.shape
        gauss = np.random.normal(0, std, (row, col, ch)).astype(np.uint8)
        noisy = cv2.add(self.processed_image, gauss)
        self.update_and_display(noisy)

    def erosion(self, kernel_size=3):
        kernel_size = int(kernel_size) | 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.update_and_display(cv2.erode(self.processed_image, kernel, iterations=1))

    def dilation(self, kernel_size=3):
        kernel_size = int(kernel_size) | 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.update_and_display(cv2.dilate(self.processed_image, kernel, iterations=1))

    def opening(self, kernel_size=3):
        kernel_size = int(kernel_size) | 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.update_and_display(cv2.morphologyEx(self.processed_image, cv2.MORPH_OPEN, kernel))

    def closing(self, kernel_size=3):
        kernel_size = int(kernel_size) | 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.update_and_display(cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel))

    def boundary_extraction(self, kernel_size=3):
        kernel_size = int(kernel_size) | 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(self.processed_image, kernel, iterations=1)
        self.update_and_display(cv2.subtract(self.processed_image, eroded))

    def region_filling(self):
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        h, w = mask.shape
        floodfill_mask = np.zeros((h+2, w+2), np.uint8)
        filled = mask.copy()
        cv2.floodFill(filled, floodfill_mask, (0, 0), 255)
        self.update_and_display(cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR))

    def global_threshold(self, thresh=127):
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        _, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        self.update_and_display(cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR))

    def adaptive_threshold(self):
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.update_and_display(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    def otsu_threshold(self):
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.update_and_display(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    def hough_transform(self, threshold=100):
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=50, maxLineGap=10)
        img = self.processed_image.copy()
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.update_and_display(img)

    def watershed_segmentation(self):
        img = self.processed_image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255, 0, 255]  # Magenta borders
        self.update_and_display(img)

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()


# In[ ]:




