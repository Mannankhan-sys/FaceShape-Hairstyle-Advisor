import cv2
import mediapipe as mp
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import numpy as np
from face_shape import classify_face_shape
from recommendations import get_recommendations

# Appearance settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Folder where images are stored
IMAGE_DIR = "gui_face_shape_images"

# --------------------
# Face Detector Class
# --------------------
class FaceDetector:
    def __init__(self):
        self.cap = None
        self.running = False
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.history = []
        self.gender = "male"
        self.shape = ""
        self.rec = ""
        self.confidence = 0
        self.current_frame = None

    def start(self, gender):
        self.gender = gender
        if not self.running:
            # Try indices 0, 1, 2 for camera
            for idx in [0, 1, 2]:
                print(f"Attempting to open camera index {idx}...", flush=True)
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    print(f"Success! Camera index {idx} opened.", flush=True)
                    self.running = True
                    threading.Thread(target=self.update_loop, daemon=True).start()
                    return True
                self.cap.release()
            
            print("Error: Could not open any camera index.", flush=True)
            return False
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_loop(self):
        print("Starting MediaPipe Thread...", flush=True)
        
        # 1. Initialize FaceMesh
        face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # 2. Immediately update the current frame for the UI
            # We clone it so we can draw on one and keep the original if needed
            display_frame = frame.copy()
            
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                # Find largest face
                largest_face = None
                max_area = 0
                for face in results.multi_face_landmarks:
                    x_coords = [lm.x for lm in face.landmark]
                    y_coords = [lm.y for lm in face.landmark]
                    area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                    if area > max_area:
                        max_area = area
                        largest_face = face

                if largest_face:
                    # Draw on display_frame
                    self.mp_draw.draw_landmarks(
                        display_frame, largest_face, self.mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 255, 100), thickness=1, circle_radius=1),
                        connection_drawing_spec=self.mp_draw.DrawingSpec(color=(0, 200, 50), thickness=1)
                    )
                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in largest_face.landmark]

                    # Logic
                    shape = classify_face_shape(landmarks)
                    self.history.append(shape)
                    if len(self.history) > 15:
                        self.history.pop(0)
                    self.shape = max(set(self.history), key=self.history.count)
                    self.confidence = round(self.history.count(self.shape) / len(self.history) * 100, 1)
                    self.rec = get_recommendations(self.shape, self.gender)
            else:
                self.shape = ""
                self.confidence = 0

            # Update the shared frame for UI
            self.current_frame = display_frame
        
        face_mesh.close()

# --------------------
# GUI Application Class
# --------------------
class App(ctk.CTk):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.last_shape = ""
        self.ui_counter = 0
        
        self.title("AI Face Shape & Hairstyle Advisor")
        self.geometry("1100x750")
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="StyleAI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(pady=(20, 30))

        self.gender_label = ctk.CTkLabel(self.sidebar, text="Gender Selection", font=ctk.CTkFont(size=14, weight="bold"))
        self.gender_label.pack(pady=(10, 5))
        
        self.gender_var = ctk.StringVar(value="male")
        self.male_radio = ctk.CTkRadioButton(self.sidebar, text="Male", variable=self.gender_var, value="male", command=self.update_gender_live)
        self.male_radio.pack(pady=5)
        self.female_radio = ctk.CTkRadioButton(self.sidebar, text="Female", variable=self.gender_var, value="female", command=self.update_gender_live)
        self.female_radio.pack(pady=5)

        self.start_btn = ctk.CTkButton(self.sidebar, text="Start Analysis", command=self.start_camera, fg_color="#2ecc71", hover_color="#27ae60")
        self.start_btn.pack(pady=(40, 10), padx=20)
        
        self.stop_btn = ctk.CTkButton(self.sidebar, text="Stop Camera", command=self.stop_camera, fg_color="#e74c3c", hover_color="#c0392b")
        self.stop_btn.pack(pady=10, padx=20)

        self.status_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.status_frame.pack(side="bottom", pady=20, padx=10)
        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Ready", font=ctk.CTkFont(size=12))
        self.status_label.pack()

        self.main_container = ctk.CTkFrame(self, corner_radius=15)
        self.main_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.cam_label = ctk.CTkLabel(self.main_container, text="Camera Preview\nPress Start to Begin", 
                                      font=ctk.CTkFont(size=16), compound="top",
                                      width=640, height=480)
        self.cam_label.pack(expand=True, fill="both", padx=10, pady=10)

        self.results_panel = ctk.CTkFrame(self, width=300, corner_radius=15)
        self.results_panel.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")
        
        self.res_header = ctk.CTkLabel(self.results_panel, text="Detection Results", font=ctk.CTkFont(size=20, weight="bold"))
        self.res_header.pack(pady=15)

        self.shape_info = ctk.CTkLabel(self.results_panel, text="Shape: -", font=ctk.CTkFont(size=18))
        self.shape_info.pack(pady=5)
        
        self.conf_bar = ctk.CTkProgressBar(self.results_panel, width=200)
        self.conf_bar.set(0)
        self.conf_bar.pack(pady=5)
        
        self.conf_text = ctk.CTkLabel(self.results_panel, text="Confidence: 0%", font=ctk.CTkFont(size=12))
        self.conf_text.pack(pady=(0, 15))

        self.rec_label = ctk.CTkLabel(self.results_panel, text="Recommendations", font=ctk.CTkFont(size=16, weight="bold"))
        self.rec_label.pack(pady=(10, 5))

        self.rec_scroll = ctk.CTkScrollableFrame(self.results_panel, label_text="Styles")
        self.rec_scroll.pack(expand=True, fill="both", padx=10, pady=10)

        self.update_gui()

    def update_gender_live(self):
        self.detector.gender = self.gender_var.get()
        # Force refresh recommendations if we have a shape
        if self.detector.shape:
            self.refresh_recommendations()

    def start_camera(self):
        gender = self.gender_var.get()
        if self.detector.start(gender):
            self.status_label.configure(text="Status: Analyzing...")
        else:
            self.status_label.configure(text="Status: Error - No Camera")

    def stop_camera(self):
        self.detector.stop()
        self.status_label.configure(text="Status: Stopped")
        self.cam_label.configure(image="", text="Camera Preview\nPress Start to Begin")

    def update_gui(self):
        if self.detector.running:
            if self.detector.current_frame is not None:
                # Debugging print to terminal every 100 frames
                if self.ui_counter % 100 == 0:
                    print(f"DEBUG: UI rendering frame {self.ui_counter}", flush=True)
                self.ui_counter += 1

                frame = self.detector.current_frame
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resize = cv2.resize(img, (640, 480))
                img_pil = Image.fromarray(img_resize)
                img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
                
                self.cam_label.configure(image=img_tk, text="")
                self.cam_label._image_ref = img_tk # Keep persistent reference
                
                if self.detector.shape:
                    self.shape_info.configure(text=f"Shape: {self.detector.shape}")
                    self.conf_bar.set(self.detector.confidence / 100)
                    self.conf_text.configure(text=f"Confidence: {self.detector.confidence}%")
                    
                    if self.detector.shape != self.last_shape:
                        self.refresh_recommendations()
                        self.last_shape = self.detector.shape
        
        self.after(20, self.update_gui)

    def refresh_recommendations(self):
        for widget in self.rec_scroll.winfo_children():
            widget.destroy()

        hair_list, beard_list = [], []
        if self.detector.gender == "male" and isinstance(self.detector.rec, tuple):
            hair_list = self.detector.rec[0].split(',')
            beard_list = self.detector.rec[1].split(',')
        else:
            hair_list = str(self.detector.rec).split(',')
            beard_list = []

        ctk.CTkLabel(self.rec_scroll, text="Hairstyles", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(5, 5))
        for h in hair_list:
            btn = ctk.CTkButton(self.rec_scroll, text=h.strip(), fg_color="transparent", border_width=1, anchor="w")
            btn.pack(fill="x", pady=2)

        if beard_list:
            ctk.CTkLabel(self.rec_scroll, text="Beards", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(15, 5))
            for b in beard_list:
                btn = ctk.CTkButton(self.rec_scroll, text=b.strip(), fg_color="transparent", border_width=1, anchor="w", border_color="#3498db")
                btn.pack(fill="x", pady=2)

    def on_close(self):
        self.detector.stop()
        self.destroy()

if __name__ == "__main__":
    detector = FaceDetector()
    app = App(detector)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
