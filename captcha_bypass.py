import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import pytesseract
from tkinter import filedialog
from tkinter import DoubleVar
import json
import os
from pathlib import Path


class LayerFrame(ttk.LabelFrame):
    def __init__(self, parent, app, layer_data):
        super().__init__(parent, text=f"Layer {layer_data['id']}")
        self.app = app
        self.layer_data = layer_data
        self.drag_data = {"x": 0, "y": 0, "item": None}
        
        # Create main frame with padding
        main_frame = ttk.Frame(self, padding="5")
        main_frame.pack(fill="x", expand=True)
        
        # Create drag handle with visual feedback
        self.drag_handle = ttk.Label(main_frame, text="⋮⋮", cursor="hand2")
        self.drag_handle.pack(side="left", padx=(0, 5))
        self.drag_handle.bind("<ButtonPress-1>", self.on_drag_start)
        self.drag_handle.bind("<B1-Motion>", self.on_drag_motion)
        self.drag_handle.bind("<ButtonRelease-1>", self.on_drag_release)
        
        # Create type selection with tooltip
        type_frame = ttk.Frame(main_frame)
        type_frame.pack(side="left", fill="x", expand=True)
        
        ttk.Label(type_frame, text="Operation:").pack(side="left")
        self.type_var = tk.StringVar(value=layer_data["type"])
        self.type_combo = ttk.Combobox(type_frame, textvariable=self.type_var, 
                                     values=["Binary", "Adaptive", "Otsu", "GaussianBlur", 
                                            "Invert", "RemoveSmallNoise", "RemoveThinLines", 
                                            "SmoothEdges", "Morphology"], state="readonly")
        self.type_combo.pack(side="left", padx=5)
        self.type_combo.bind("<<ComboboxSelected>>", self.on_type_change)
        
        # Add tooltip for operation types
        self.create_tooltip(self.type_combo, {
            "Binary": "Simple thresholding operation. Pixels above threshold become white, below become black.",
            "Adaptive": "Adaptive thresholding. Applies thresholding based on local region.",
            "Otsu": "Automatic thresholding. Finds optimal threshold value based on image histogram.",
            "GaussianBlur": "Gaussian blurring. Used for noise reduction and smoothing.",
            "Invert": "Inverts the image colors.",
            "RemoveSmallNoise": "Cleans small noise points.",
            "RemoveThinLines": "Removes thin lines.",
            "SmoothEdges": "Smooths and preserves edges.",
            "Morphology": "Morphological operations. Opening and closing operations."
        })
        
        # Create value controls with tooltips
        self.value_frame = ttk.Frame(main_frame)
        self.value_frame.pack(side="left", fill="x", expand=True)
        
        self.value_label = ttk.Label(self.value_frame, text="Value:")
        self.value_label.pack(side="left")
        self.value_var = tk.DoubleVar(value=layer_data["value"])
        self.value_scale = tk.Scale(self.value_frame, from_=0, to=1, 
                                   variable=self.value_var, orient="horizontal",
                                   resolution=0.1, command=lambda x: self.on_value_change())
        self.value_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        # Create second value frame with tooltip
        self.second_value_frame = ttk.Frame(main_frame)
        self.second_value_frame.pack(side="left", fill="x", expand=True)
        
        self.second_value_label = ttk.Label(self.second_value_frame, text="Second Value:")
        self.second_value_label.pack(side="left")
        self.second_value_var = tk.DoubleVar(value=layer_data["second_value"])
        self.second_value_scale = tk.Scale(self.second_value_frame, from_=0, to=1,
                                          variable=self.second_value_var, orient="horizontal",
                                          resolution=0.1, command=lambda x: self.on_value_change())
        self.second_value_scale.pack(side="left", fill="x", expand=True, padx=5)
        
        # Create delete button with tooltip
        self.delete_button = ttk.Button(main_frame, text="×", width=3,
                                      command=self.delete_layer)
        self.delete_button.pack(side="right", padx=(5, 0))
        self.create_tooltip(self.delete_button, "Delete this layer")
        
        # Initialize tooltips for value controls
        self.update_value_tooltips()
        
        # Set initial state
        self.on_type_change()
    
    def create_tooltip(self, widget, text_dict):
        """Create a tooltip for a widget"""
        def show_tooltip(event):
            if isinstance(text_dict, dict):
                text = text_dict.get(self.type_var.get(), "")
            else:
                text = text_dict
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, justify="left",
                            background="#ffffe0", relief="solid", borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
            
            widget.tooltip = tooltip
            widget.bind("<Leave>", lambda e: hide_tooltip())
            tooltip.bind("<Leave>", lambda e: hide_tooltip())
        
        widget.bind("<Enter>", show_tooltip)
    
    def update_value_tooltips(self):
        """Update tooltips for value controls based on operation type"""
        op_type = self.type_var.get()
        
        # Update primary value tooltip
        value_text = f"Value: {self.value_var.get():.1f}"
        if op_type == "Binary":
            value_text += " (0-255)"
        elif op_type in ["Adaptive", "GaussianBlur", "RemoveSmallNoise", "RemoveThinLines", "SmoothEdges", "Morphology"]:
            value_text += " (Kernel Size)"
        self.create_tooltip(self.value_scale, value_text)
        
        # Update secondary value tooltip
        second_value_text = f"Second Value: {self.second_value_var.get():.1f}"
        if op_type == "Adaptive":
            second_value_text += " (C Value: -50-50)"
        elif op_type in ["GaussianBlur", "SmoothEdges"]:
            second_value_text += " (Sigma: 0.1-3.0)"
        elif op_type in ["RemoveSmallNoise", "RemoveThinLines"]:
            second_value_text += " (Iterations: 1-5)"
        elif op_type == "Morphology":
            second_value_text += " (0: Opening, 1: Closing)"
        self.create_tooltip(self.second_value_scale, second_value_text)
    
    def on_drag_start(self, event):
        """Start dragging the layer"""
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.drag_data["item"] = self
        self.drag_handle.configure(foreground="blue")
    
    def on_drag_motion(self, event):
        """Handle layer dragging"""
        if self.drag_data["item"]:
            # Calculate movement
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            # Get current position
            x = self.winfo_x() + dx
            y = self.winfo_y() + dy
            
            # Move the layer
            self.place(x=x, y=y)
    
    def on_drag_release(self, event):
        """Handle end of drag operation"""
        if self.drag_data["item"]:
            self.drag_handle.configure(foreground="black")
            self.drag_data["item"] = None
            self.app.reorder_layers()
    
    def reorder_layers(self):
        """Reorder layers based on their vertical position"""
        layers = self.app.layers
        layers.sort(key=lambda x: x.winfo_y())
        for i, layer in enumerate(layers):
            layer.configure(text=f"Layer {i+1}")
            layer.layer_data["id"] = i+1

    def on_type_change(self, event=None):
        """Handle operation type changes"""
        op_type = self.type_var.get()
        
        # Store current values
        current_value = self.value_var.get()
        current_second_value = self.second_value_var.get()
        
        # Configure scales based on operation type
        if op_type == "Binary":
            self.value_label.config(text="Threshold Value")
            self.value_scale.config(from_=0, to=255, resolution=1.0)
            self.value_var.set(min(255, max(0, current_value)))
            self.second_value_frame.pack_forget()
            
        elif op_type == "Adaptive":
            self.value_label.config(text="Block Size")
            self.value_scale.config(from_=3, to=99, resolution=1.0)
            self.value_var.set(min(99, max(3, current_value)))
            self.second_value_label.config(text="C Value")
            self.second_value_scale.config(from_=-50, to=50, resolution=1.0)
            self.second_value_var.set(min(50, max(-50, current_second_value)))
            self.second_value_frame.pack(side="left", fill="x", expand=True)
            
        elif op_type == "GaussianBlur":
            self.value_label.config(text="Kernel Size")
            self.value_scale.config(from_=1, to=15, resolution=1.0)
            self.value_var.set(min(15, max(1, current_value)))
            self.second_value_label.config(text="Sigma")
            self.second_value_scale.config(from_=0.1, to=3.0, resolution=0.1)
            self.second_value_var.set(min(3.0, max(0.1, current_second_value)))
            self.second_value_frame.pack(side="left", fill="x", expand=True)
            
        elif op_type == "RemoveSmallNoise":
            self.value_label.config(text="Kernel Size")
            self.value_scale.config(from_=1, to=5, resolution=1.0)
            self.value_var.set(min(5, max(1, current_value)))
            self.second_value_label.config(text="Iterations")
            self.second_value_scale.config(from_=1, to=5, resolution=1.0)
            self.second_value_var.set(min(5, max(1, current_second_value)))
            self.second_value_frame.pack(side="left", fill="x", expand=True)
            
        elif op_type == "RemoveThinLines":
            self.value_label.config(text="Kernel Size")
            self.value_scale.config(from_=1, to=5, resolution=1.0)
            self.value_var.set(min(5, max(1, current_value)))
            self.second_value_label.config(text="Iterations")
            self.second_value_scale.config(from_=1, to=5, resolution=1.0)
            self.second_value_var.set(min(5, max(1, current_second_value)))
            self.second_value_frame.pack(side="left", fill="x", expand=True)
            
        elif op_type == "SmoothEdges":
            self.value_label.config(text="Diameter")
            self.value_scale.config(from_=1, to=15, resolution=1.0)
            self.value_var.set(min(15, max(1, current_value)))
            self.second_value_label.config(text="Sigma")
            self.second_value_scale.config(from_=0.1, to=3.0, resolution=0.1)
            self.second_value_var.set(min(3.0, max(0.1, current_second_value)))
            self.second_value_frame.pack(side="left", fill="x", expand=True)
            
        elif op_type == "Morphology":
            self.value_label.config(text="Kernel Size")
            self.value_scale.config(from_=1, to=5, resolution=1.0)
            self.value_var.set(min(5, max(1, current_value)))
            self.second_value_label.config(text="Operation Type")
            self.second_value_scale.config(from_=0, to=1, resolution=1.0)
            self.second_value_var.set(min(1, max(0, current_second_value)))
            self.second_value_frame.pack(side="left", fill="x", expand=True)
            
        else:  # Otsu or Invert
            self.value_frame.pack_forget()
            self.second_value_frame.pack_forget()
        
        # Update layer data
        self.layer_data["type"] = op_type
        self.layer_data["value"] = self.value_var.get()
        self.layer_data["second_value"] = self.second_value_var.get()
        
        # Update tooltips
        self.update_value_tooltips()
        
        # Update image
        self.app.update_image()
        
        # Force update of the UI
        self.app.root.update_idletasks()

    def get_layer(self):
        return {
            "type": self.type_var.get(),
            "value": self.value_var.get(),
            "second_value": self.second_value_var.get()
        }

    def delete_layer(self):
        self.app.remove_layer(self.layer_data["id"])

    def on_value_change(self, event=None):
        """Handle value changes in scales"""
        try:
            # Get current values
            value = self.value_var.get()
            second_value = self.second_value_var.get()
            
            # Update layer data
            self.layer_data["value"] = value
            self.layer_data["second_value"] = second_value
            
            # Update tooltips
            self.update_value_tooltips()
            
            # Update image
            self.app.update_image()
            
            # Force update of the UI
            self.app.root.update_idletasks()
            
        except Exception as e:
            print(f"Error in value change: {str(e)}")
            # Reset to previous valid values if error occurs
            self.value_var.set(self.layer_data["value"])
            self.second_value_var.set(self.layer_data["second_value"])


class CaptchaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Captcha Bypass Tool")

        self.original_img = None
        self.processed_img = None
        self.layers = []
        self.preview_size = (300, 100)
        
        # Create presets directory if it doesn't exist
        self.presets_dir = Path("presets")
        self.presets_dir.mkdir(exist_ok=True)
        
        # Last used preset
        self.last_used_preset = None

        # Main layout
        self.create_layout()
        
        # Create menu bar
        self.create_menu()
        
        # Check Tesseract installation
        self.check_tesseract()

    def create_layout(self):
        # Image preview frames
        preview_frame = ttk.Frame(self.root)
        preview_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        # Original image preview
        self.original_canvas = tk.Label(preview_frame)
        self.original_canvas.grid(row=0, column=0, padx=5)
        ttk.Label(preview_frame, text="Original").grid(row=1, column=0)

        # Processed image preview
        self.processed_canvas = tk.Label(preview_frame)
        self.processed_canvas.grid(row=0, column=1, padx=5)
        ttk.Label(preview_frame, text="Processed").grid(row=1, column=1)

        # Control buttons
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        ttk.Button(control_frame, text="📂 Load Image", command=self.load_image).pack(side="left", padx=5)
        ttk.Button(control_frame, text="➕ Add Layer", command=self.add_layer).pack(side="left", padx=5)
        ttk.Button(control_frame, text="💾 Save", command=self.save_layers).pack(side="left", padx=5)
        ttk.Button(control_frame, text="📂 Load", command=self.load_layers).pack(side="left", padx=5)
        ttk.Button(control_frame, text="🧠 Solve OCR", command=self.perform_ocr).pack(side="left", padx=5)
        ttk.Button(control_frame, text="🔄 Reset", command=self.reset_layers).pack(side="left", padx=5)

        # Layer container
        self.layer_container = tk.Frame(self.root)
        self.layer_container.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

        # OCR result
        result_frame = ttk.Frame(self.root)
        result_frame.grid(row=3, column=0, columnspan=3, pady=5)
        
        self.ocr_result = tk.StringVar()
        tk.Label(result_frame, textvariable=self.ocr_result, font=("Courier", 14), fg="blue").pack(side="left")
        ttk.Button(result_frame, text="📋 Copy", command=self.copy_ocr_result).pack(side="left", padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky="ew", padx=10, pady=5)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Presets menu
        presets_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=presets_menu)
        presets_menu.add_command(label="Save Current Settings", command=self.save_preset)
        presets_menu.add_command(label="Load Settings", command=self.load_preset)
        presets_menu.add_separator()
        presets_menu.add_command(label="Load Default Settings", command=self.load_default_preset)
        presets_menu.add_command(label="Load Last Used Settings", command=self.load_last_preset)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)

    def save_preset(self):
        if not self.layers:
            messagebox.showwarning("Warning", "No settings to save!")
            return

        preset_name = simpledialog.askstring("Save Settings", 
                                           "Enter a name for the settings:",
                                           initialvalue="new_preset")
        if not preset_name:
            return

        try:
            preset_path = self.presets_dir / f"{preset_name}.json"
            # Ensure all values are properly formatted before saving
            layers_data = []
            for layer in self.layers:
                layer_data = {
                    "id": layer.layer_data["id"],
                    "type": layer.layer_data["type"],
                    "value": float(layer.layer_data["value"]),  # Ensure value is float
                    "second_value": float(layer.layer_data["second_value"])  # Ensure second_value is float
                }
                layers_data.append(layer_data)

            with open(preset_path, 'w') as f:
                json.dump(layers_data, f, indent=2)
            
            self.last_used_preset = preset_name
            messagebox.showinfo("Success", f"Settings saved as '{preset_name}'!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving settings: {str(e)}")

    def load_preset(self):
        preset_files = list(self.presets_dir.glob("*.json"))
        if not preset_files:
            messagebox.showwarning("Warning", "No saved settings found!")
            return

        preset_names = [f.stem for f in preset_files]
        preset_name = simpledialog.askstring("Load Settings",
                                           "Select settings to load:",
                                           initialvalue=self.last_used_preset or preset_names[0])
        if not preset_name:
            return

        try:
            preset_path = self.presets_dir / f"{preset_name}.json"
            if not preset_path.exists():
                messagebox.showerror("Error", f"Settings '{preset_name}' not found!")
                return

            with open(preset_path, 'r') as f:
                layers_data = json.load(f)

            # Validate and convert data before creating layers
            for layer_data in layers_data:
                # Ensure all required fields exist
                required_fields = ["id", "type", "value", "second_value"]
                if not all(field in layer_data for field in required_fields):
                    raise ValueError("Invalid layer data format")

                # Convert values to proper types
                layer_data["id"] = int(layer_data["id"])
                layer_data["value"] = float(layer_data["value"])
                layer_data["second_value"] = float(layer_data["second_value"])

                # Validate operation type
                valid_types = ["Binary", "Adaptive", "Otsu", "GaussianBlur", 
                             "Invert", "RemoveSmallNoise", "RemoveThinLines", 
                             "SmoothEdges", "Morphology"]
                if layer_data["type"] not in valid_types:
                    raise ValueError(f"Invalid operation type: {layer_data['type']}")

            # Clear existing layers
            for layer in self.layers:
                layer.destroy()
            self.layers.clear()

            # Create new layers with validated data
            for i, layer_data in enumerate(layers_data):
                frame = LayerFrame(self.layer_container, self, layer_data)
                frame.grid(row=i, column=0, pady=5, sticky="ew")
                self.layers.append(frame)

            self.last_used_preset = preset_name
            self.update_image()
            messagebox.showinfo("Success", f"Settings '{preset_name}' loaded!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading settings: {str(e)}")

    def load_default_preset(self):
        default_preset = self.presets_dir / "default.json"
        if not default_preset.exists():
            messagebox.showwarning("Warning", "Default settings not found!")
            return

        try:
            with open(default_preset, 'r') as f:
                layers_data = json.load(f)

            # Clear existing layers
            for layer in self.layers:
                layer.destroy()
            self.layers.clear()

            # Create new layers
            for i, layer_data in enumerate(layers_data):
                frame = LayerFrame(self.layer_container, self, layer_data)
                frame.grid(row=i, column=0, pady=5, sticky="ew")
                self.layers.append(frame)

            self.last_used_preset = "default"
            self.update_image()
            messagebox.showinfo("Success", "Default settings loaded!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading default settings: {str(e)}")

    def load_last_preset(self):
        if not self.last_used_preset:
            messagebox.showwarning("Warning", "No last used settings found!")
            return

        preset_path = self.presets_dir / f"{self.last_used_preset}.json"
        if not preset_path.exists():
            messagebox.showerror("Error", "Last used settings not found!")
            return

        try:
            with open(preset_path, 'r') as f:
                layers_data = json.load(f)

            # Clear existing layers
            for layer in self.layers:
                layer.destroy()
            self.layers.clear()

            # Create new layers
            for i, layer_data in enumerate(layers_data):
                frame = LayerFrame(self.layer_container, self, layer_data)
                frame.grid(row=i, column=0, pady=5, sticky="ew")
                self.layers.append(frame)

            self.update_image()
            messagebox.showinfo("Success", "Last used settings loaded!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading last used settings: {str(e)}")

    def show_help(self):
        help_text = """
        🔍 Captcha Bypass Tool User Guide

        1. Loading Images:
           - Click "📂 Load Image" to select a captcha image
           - Supported formats: JPG, JPEG, PNG

        2. Layer Operations:
           - Add a new processing layer with "➕ Add Layer"
           - Configure operation type and parameters for each layer
           - Reorder layers using drag-and-drop interface
           - Delete layers using the "×" button

        3. Settings Management:
           - Save current settings from the settings menu
           - Use saved settings with other images
           - Load default settings
           - Quickly load last used settings

        4. OCR Process:
           - Start text recognition with "🧠 Solve OCR"
           - Results are displayed automatically
           - Copy results to clipboard with "📋 Copy"

        5. Operation Types:
           - Binary: Simple thresholding
           - Adaptive: Adaptive thresholding
           - Otsu: Automatic thresholding
           - GaussianBlur: Blurring
           - RemoveSmallNoise: Noise removal
           - RemoveThinLines: Thin line removal
           - SmoothEdges: Edge smoothing
           - Morphology: Morphological operations
        """
        messagebox.showinfo("User Guide", help_text)

    def show_about(self):
        about_text = """
        🔍 Captcha Bypass Tool
        Version 1.0

        This application is designed to facilitate
        automated captcha solving using advanced
        image processing techniques and OCR.

        Features:
        - Multi-layer support
        - Drag-and-drop interface
        - Settings management
        - Tesseract OCR integration
        """
        messagebox.showinfo("About", about_text)

    def find_layer_position(self, layer_frame):
        # Find the new position based on the layer's current position
        y_pos = layer_frame.winfo_y()
        new_pos = 0
        for i, layer in enumerate(self.layers):
            if layer.winfo_y() < y_pos:
                new_pos = i + 1
        return min(new_pos, len(self.layers) - 1)

    def reorder_layer(self, old_pos, new_pos):
        if old_pos == new_pos:
            return

        # Remove layer from old position
        layer = self.layers.pop(old_pos)
        
        # Insert at new position
        self.layers.insert(new_pos, layer)
        
        # Update layer IDs
        for i, layer in enumerate(self.layers):
            layer.layer_data["id"] = i
            layer.grid(row=i, column=0, pady=5, sticky="ew")
        
        self.update_image()

    def reset_layers(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to reset all layers?"):
            for layer in self.layers:
                layer.destroy()
            self.layers.clear()
            self.update_image()

    def check_tesseract(self):
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            messagebox.showerror("Error", "Tesseract OCR is not installed!\nPlease run 'brew install tesseract' command.")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        try:
            self.original_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if self.original_img is None:
                raise Exception("Failed to load image")
            
            self.update_preview()
            self.layers.clear()
            self.layer_container.destroy()
            self.layer_container = tk.Frame(self.root)
            self.layer_container.grid(row=2, column=0, columnspan=3)
            self.update_image()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def update_preview(self):
        if self.original_img is not None:
            # Original image preview
            orig_img = Image.fromarray(self.original_img).resize(self.preview_size)
            orig_tk = ImageTk.PhotoImage(orig_img)
            self.original_canvas.configure(image=orig_tk)
            self.original_canvas.image = orig_tk

            # Processed image preview
            if self.processed_img is not None:
                proc_img = Image.fromarray(self.processed_img).resize(self.preview_size)
                proc_tk = ImageTk.PhotoImage(proc_img)
                self.processed_canvas.configure(image=proc_tk)
                self.processed_canvas.image = proc_tk

    def add_layer(self):
        idx = len(self.layers)
        frame = LayerFrame(self.layer_container, self, {"id": idx, "type": "Binary", "value": 127.0, "second_value": 0.0})
        frame.grid(row=idx, column=0, pady=5, sticky="ew")
        self.layers.append(frame)
        self.update_image()

    def copy_layer(self, idx):
        if 0 <= idx < len(self.layers):
            new_idx = len(self.layers)
            frame = LayerFrame(self.layer_container, self, {"id": new_idx, "type": self.layers[idx].layer_data["type"], "value": self.layers[idx].layer_data["value"], "second_value": self.layers[idx].layer_data["second_value"]})
            frame.grid(row=new_idx, column=0, pady=5, sticky="ew")
            
            self.layers.append(frame)
            self.update_image()

    def remove_layer(self, idx):
        self.layers[idx].destroy()
        del self.layers[idx]
        for i, frame in enumerate(self.layers):
            frame.layer_data["id"] = i
        self.update_image()

    def save_layers(self):
        if not self.layers:
            messagebox.showwarning("Warning", "No layers to save!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        try:
            # Ensure all values are properly formatted before saving
            layers_data = []
            for layer in self.layers:
                layer_data = {
                    "id": layer.layer_data["id"],
                    "type": layer.layer_data["type"],
                    "value": float(layer.layer_data["value"]),  # Ensure value is float
                    "second_value": float(layer.layer_data["second_value"])  # Ensure second_value is float
                }
                layers_data.append(layer_data)

            with open(file_path, 'w') as f:
                json.dump(layers_data, f, indent=2)
            messagebox.showinfo("Success", "Layers saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving layers: {str(e)}")

    def load_layers(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                layers_data = json.load(f)

            # Validate and convert data before creating layers
            for layer_data in layers_data:
                # Ensure all required fields exist
                required_fields = ["id", "type", "value", "second_value"]
                if not all(field in layer_data for field in required_fields):
                    raise ValueError("Invalid layer data format")

                # Convert values to proper types
                layer_data["id"] = int(layer_data["id"])
                layer_data["value"] = float(layer_data["value"])
                layer_data["second_value"] = float(layer_data["second_value"])

                # Validate operation type
                valid_types = ["Binary", "Adaptive", "Otsu", "GaussianBlur", 
                             "Invert", "RemoveSmallNoise", "RemoveThinLines", 
                             "SmoothEdges", "Morphology"]
                if layer_data["type"] not in valid_types:
                    raise ValueError(f"Invalid operation type: {layer_data['type']}")

            # Clear existing layers
            for layer in self.layers:
                layer.destroy()
            self.layers.clear()

            # Create new layers with validated data
            for i, layer_data in enumerate(layers_data):
                frame = LayerFrame(self.layer_container, self, layer_data)
                frame.grid(row=i, column=0, pady=5, sticky="ew")
                self.layers.append(frame)

            self.update_image()
            messagebox.showinfo("Success", "Layers loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading layers: {str(e)}")

    def apply_layers(self, img):
        result = img.copy()
        self.progress["maximum"] = len(self.layers)
        self.progress["value"] = 0

        for i, layer in enumerate(self.layers):
            l = layer.layer_data
            try:
                if l["type"] == "Binary":
                    # Binary thresholding with improved range
                    threshold = max(0, min(255, l["value"]))
                    _, result = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)
                
                elif l["type"] == "Adaptive":
                    # Adaptive thresholding with improved parameters
                    block_size = max(3, min(99, int(l["value"])))
                    if block_size % 2 == 0:
                        block_size += 1
                    c_value = max(-50, min(50, int(l["second_value"])))
                    result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, block_size, c_value)
                
                elif l["type"] == "Otsu":
                    # Otsu's method with preprocessing
                    blur = cv2.GaussianBlur(result, (3, 3), 0)
                    _, result = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                elif l["type"] == "GaussianBlur":
                    # Gaussian blur with improved parameters
                    kernel_size = max(3, min(31, int(l["value"] * 2 + 1)))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    # Scale sigma value for more visible effect
                    sigma = max(0.1, min(3.0, l["second_value"])) * 2.0
                    result = cv2.GaussianBlur(result, (kernel_size, kernel_size), sigma)
                
                elif l["type"] == "Invert":
                    # Simple inversion
                    result = cv2.bitwise_not(result)
                
                elif l["type"] == "RemoveSmallNoise":
                    # Improved noise removal with adaptive kernel
                    kernel_size = max(3, min(7, int(l["value"] * 2 + 1)))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    iterations = max(1, min(5, int(l["second_value"])))
                    
                    # Apply opening operation with multiple iterations
                    for _ in range(iterations):
                        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                
                elif l["type"] == "RemoveThinLines":
                    # Improved thin line removal with adaptive kernel
                    kernel_size = max(3, min(7, int(l["value"] * 2 + 1)))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    iterations = max(1, min(5, int(l["second_value"])))
                    
                    # Apply closing operation with multiple iterations
                    for _ in range(iterations):
                        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
                
                elif l["type"] == "SmoothEdges":
                    # Improved edge smoothing with bilateral filter
                    d = max(3, min(15, int(l["value"] * 2 + 1)))
                    if d % 2 == 0:
                        d += 1
                    sigma_color = max(0.1, min(5.0, l["second_value"]))
                    sigma_space = max(0.1, min(5.0, l["second_value"]))
                    
                    # Apply bilateral filter with improved parameters
                    result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
                    
                    # Additional edge preservation
                    edges = cv2.Canny(result, 100, 200)
                    result = cv2.addWeighted(result, 0.7, cv2.bitwise_not(edges), 0.3, 0)
                
                elif l["type"] == "Morphology":
                    # Advanced morphological operations
                    kernel_size = max(3, min(7, int(l["value"] * 2 + 1)))
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    
                    if l["second_value"] < 0.5:
                        # Opening operation for noise removal
                        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
                    else:
                        # Closing operation for filling gaps
                        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
                    
                    # Additional refinement
                    result = cv2.medianBlur(result, 3)

                self.progress["value"] = i + 1
                self.root.update_idletasks()

            except Exception as e:
                error_msg = f"Layer {i+1} processed with error: {str(e)}"
                print(error_msg)
                try:
                    messagebox.showerror("Error", error_msg)
                except:
                    pass
                return result

        return result

    def update_image(self):
        if self.original_img is None:
            return
        self.processed_img = self.apply_layers(self.original_img)
        self.update_preview()

    def perform_ocr(self):
        if self.processed_img is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        try:
            config = '--oem 3 --psm 7'
            text = pytesseract.image_to_string(self.processed_img, config=config)
            self.ocr_result.set(f"OCR: {text.strip()}")
        except Exception as e:
            messagebox.showerror("Error", f"Error during OCR process: {str(e)}")

    def copy_ocr_result(self):
        if self.ocr_result.get():
            self.root.clipboard_clear()
            self.root.clipboard_append(self.ocr_result.get().replace("OCR: ", ""))
            messagebox.showinfo("Success", "OCR result copied to clipboard!")

    def reorder_layers(self):
        """Reorder layers based on their vertical position"""
        # Sort layers based on their y position
        self.layers.sort(key=lambda x: x.winfo_y())
        
        # Update layer positions and IDs
        for i, layer in enumerate(self.layers):
            layer.grid(row=i, column=0, pady=5, sticky="ew")
            layer.configure(text=f"Layer {i+1}")
            layer.layer_data["id"] = i+1
        
        # Update the image to reflect the new layer order
        self.update_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = CaptchaApp(root)
    root.mainloop()