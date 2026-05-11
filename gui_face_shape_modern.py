import cv2
import mediapipe as mp
import threading
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import numpy as np
from face_shape import classify_face_shape
from recommendations import get_recommendations
from face_symmetry import calculate_symmetry_score, analyze_skin_tone

# Modern Fashion UI Theme
COLORS = {
    "bg": "#0F172A",
    "sidebar": "#1E293B",
    "accent": "#818CF8",
    "accent_hover": "#6D7CF0",
    "text": "#F8FAFC",
    "card": "#334155",
    "secondary": "#94A3B8"
}

ctk.set_appearance_mode("Dark")

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
        self.symmetry_score = 0
        self.symmetry_label = ""
        self.symmetry_buffer = [] 
        self.locked = False # Unified locking
        self.skin_tone = "Neutral"
        self.skin_color = "#FFFFFF"
        self.skin_palette = "Standard"
        self.current_frame = None

    def start(self, gender):
        self.gender = gender
        if not self.running:
            # Try indices 0, 1, 2 for camera
            for idx in [0, 1, 2]:
                print(f"[{idx}] Probing camera...", flush=True)
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret:
                        print(f"[{idx}] Success! Valid frame captured.", flush=True)
                        self.running = True
                        threading.Thread(target=self.update_loop, daemon=True).start()
                        return True
                    else:
                        print(f"[{idx}] Opened but failed to capture frame.", flush=True)
                self.cap.release()
            
            print("Error: All camera probes failed.", flush=True)
            return False
        return True

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_loop(self):
        print("DEBUG: Entering MediaPipe Thread...", flush=True)
        try:
            face_mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                refine_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6
            )
            print("DEBUG: MediaPipe FaceMesh Initialized.", flush=True)
        except Exception as e:
            print(f"CRITICAL ERROR: FaceMesh fail: {e}", flush=True)
            self.running = False
            return
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("DEBUG: Lost frame in loop...", flush=True)
                continue

            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # 2. Immediately update the current frame for the UI
            # We clone it so we can draw on one and keep the original if needed
            self.original_frame = frame.copy()
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
                    # 1. Subtle Background Tessellation (Soft Lavender)
                    self.mp_draw.draw_landmarks(
                        display_frame, largest_face, self.mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(color=(230, 216, 173), thickness=1)
                    )
                    
                    # 2. Key Contours (Eyes, Lips, Face Oval) - Neon Rose
                    self.mp_draw.draw_landmarks(
                        display_frame, largest_face, self.mp_face.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(color=(200, 100, 255), thickness=1)
                    )

                    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in largest_face.landmark]

                    # 3. Highlight key measurement points with localized glow
                    # Indices: 10=Forehead, 152=Chin, 234=L-Cheek, 454=R-Cheek, 1=Nose
                    for idx in [10, 152, 234, 454, 1]:
                        if idx < len(landmarks):
                            pt = landmarks[idx]
                            # Glow Layers
                            cv2.circle(display_frame, pt, 8, (255, 100, 200), 1)  # Outer subtle ring
                            cv2.circle(display_frame, pt, 5, (255, 255, 255), -1) # Bright center
                            cv2.circle(display_frame, pt, 6, (200, 100, 255), 1)  # Inner neon ring

                    if not self.locked:
                        # 1. Shape Detection
                        raw_shape = classify_face_shape(landmarks)
                        self.history.append(raw_shape)
                        if len(self.history) > 20: self.history.pop(0)
                        self.shape = max(set(self.history), key=self.history.count)
                        self.confidence = round(self.history.count(self.shape) / len(self.history) * 100, 1)
                        self.rec = get_recommendations(self.shape, self.gender)

                        # 2. Symmetry Analysis
                        s_score, s_label = calculate_symmetry_score(landmarks)
                        if s_score > 0:
                            self.symmetry_buffer.append((s_score, s_label))
                        
                        # 3. Skin Tone Analysis
                        self.skin_tone, self.skin_color, self.skin_palette = analyze_skin_tone(frame, landmarks)

                        # Check for Stability and Lock
                        if len(self.symmetry_buffer) >= 25 and len(self.history) >= 20:
                            avg_score = sum(s[0] for s in self.symmetry_buffer) / len(self.symmetry_buffer)
                            self.symmetry_score = round(avg_score, 1)
                            self.symmetry_label = s_label
                            self.locked = True
                    # If locked, metrics freeze to provide a formal analysis result
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
        
        self.title("Aesthetic AI | Face Shape & Style Advisor")
        self.geometry("1200x820")
        self.configure(fg_color=COLORS["bg"])
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=COLORS["sidebar"], border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="✨Aesthetic AI", font=ctk.CTkFont(size=28, weight="bold"), text_color=COLORS["accent"])
        self.logo_label.pack(pady=(20, 30))

        self.gender_label = ctk.CTkLabel(self.sidebar, text="Gender Selection", font=ctk.CTkFont(size=14, weight="bold"))
        self.gender_label.pack(pady=(10, 5))
        
        self.gender_var = ctk.StringVar(value="male")
        self.male_radio = ctk.CTkRadioButton(self.sidebar, text="Male", variable=self.gender_var, value="male", 
                                             command=self.update_gender_live, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"])
        self.male_radio.pack(pady=5)
        self.female_radio = ctk.CTkRadioButton(self.sidebar, text="Female", variable=self.gender_var, value="female", 
                                               command=self.update_gender_live, fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"])
        self.female_radio.pack(pady=5)

        self.start_btn = ctk.CTkButton(self.sidebar, text="Analysis Scan", command=self.start_camera, 
                                       fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"], height=40, font=ctk.CTkFont(weight="bold"))
        self.start_btn.pack(pady=(40, 10), padx=30)
        
        self.stop_btn = ctk.CTkButton(self.sidebar, text="Terminate Feed", command=self.stop_camera, 
                                      fg_color="transparent", border_width=1, border_color="#ef4444", text_color="#ef4444", hover_color="#7f1d1d")
        self.stop_btn.pack(pady=10, padx=30)

        self.reset_btn = ctk.CTkButton(self.sidebar, text="🔄 Retake Analysis", command=self.reset_analysis,
                                           fg_color="#2ecc71", hover_color="#27ae60", text_color="white", height=40, font=ctk.CTkFont(weight="bold"))
        self.reset_btn.pack(pady=20, padx=30)

        self.status_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.status_frame.pack(side="bottom", pady=20, padx=10)
        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Ready", font=ctk.CTkFont(size=12))
        self.status_label.pack()

        self.main_container = ctk.CTkFrame(self, corner_radius=20, fg_color=COLORS["card"])
        self.main_container.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.cam_label = ctk.CTkLabel(self.main_container, text="Camera Preview\nPress Start to Begin", 
                                      font=ctk.CTkFont(size=18, slant="italic"), compound="top",
                                      width=720, height=540, text_color=COLORS["secondary"])
        self.cam_label.pack(expand=True, fill="both", padx=15, pady=15)

        self.results_panel = ctk.CTkFrame(self, width=320, corner_radius=20, fg_color=COLORS["sidebar"])
        self.results_panel.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")
        
        # Header Badge
        self.status_badge = ctk.CTkLabel(self.results_panel, text="[ INITIALIZING ]", font=ctk.CTkFont(size=11, weight="bold"), 
                                          fg_color=COLORS["card"], corner_radius=5)
        self.status_badge.pack(pady=(20, 10), padx=20, fill="x")

        # 1. Face Shape Block
        self.shape_frame = ctk.CTkFrame(self.results_panel, fg_color="transparent")
        self.shape_frame.pack(fill="x", padx=15, pady=5)
        
        self.shape_info = ctk.CTkLabel(self.shape_frame, text="FACIAL GEOMETRY", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["accent"])
        self.shape_info.pack(anchor="w")
        
        self.shape_val = ctk.CTkLabel(self.shape_frame, text="-", font=ctk.CTkFont(size=18, weight="bold"))
        self.shape_val.pack(anchor="w", pady=(0, 5))
        
        self.conf_bar = ctk.CTkProgressBar(self.shape_frame, width=280, height=6, fg_color=COLORS["card"], progress_color=COLORS["accent"])
        self.conf_bar.set(0)
        self.conf_bar.pack(pady=2)
        
        # 2. Symmetry Block
        self.sym_frame = ctk.CTkFrame(self.results_panel, fg_color="transparent")
        self.sym_frame.pack(fill="x", padx=15, pady=10)
        
        self.sym_header = ctk.CTkLabel(self.sym_frame, text="SYMMETRY INDEX", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["accent"])
        self.sym_header.pack(anchor="w")
        
        self.sym_val = ctk.CTkLabel(self.sym_frame, text="0.0%", font=ctk.CTkFont(size=18, weight="bold"))
        self.sym_val.pack(anchor="w")
        
        self.sym_bar = ctk.CTkProgressBar(self.sym_frame, width=280, height=6, fg_color=COLORS["card"], progress_color="#2ecc71")
        self.sym_bar.set(0)
        self.sym_bar.pack(pady=2)
        
        self.sym_desc = ctk.CTkLabel(self.sym_frame, text="Analysis Pending", font=ctk.CTkFont(size=11, slant="italic"), text_color=COLORS["secondary"])
        self.sym_desc.pack(anchor="w")

        # 3. Skin Analysis Block
        self.skin_frame = ctk.CTkFrame(self.results_panel, fg_color="transparent")
        self.skin_frame.pack(fill="x", padx=15, pady=10)
        
        self.skin_header = ctk.CTkLabel(self.skin_frame, text="COLOR PALETTE", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["accent"])
        self.skin_header.pack(anchor="w")
        
        self.skin_val = ctk.CTkLabel(self.skin_frame, text="-", font=ctk.CTkFont(size=15, weight="bold"))
        self.skin_val.pack(anchor="w")
        
        self.palette_val = ctk.CTkLabel(self.skin_frame, text="Awaiting Scan", font=ctk.CTkFont(size=11), text_color=COLORS["secondary"])
        self.palette_val.pack(anchor="w")

        # 4. Style Block
        self.rec_label = ctk.CTkLabel(self.results_panel, text="ENGINEERED CURATIONS", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["accent"])
        self.rec_label.pack(anchor="w", padx=15, pady=(10, 0))

        self.rec_container = ctk.CTkFrame(self.results_panel, fg_color="transparent")
        self.rec_container.pack(expand=True, fill="both", padx=10, pady=5)

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

    def reset_analysis(self):
        self.detector.history = []
        self.detector.symmetry_buffer = []
        self.detector.locked = False
        self.detector.shape = ""
        self.detector.symmetry_score = 0
        self.detector.symmetry_label = "Analyzing..."
        self.detector.skin_tone = "Analyzing..."
        self.status_label.configure(text="Status: Scan Reset")
        self.status_badge.configure(text="[ SCANNING LIVE ]", text_color=COLORS["accent"])

    def update_gui(self):
        if self.detector.running:
            if self.detector.current_frame is not None:
                # Debugging print to terminal every 100 frames
                if self.ui_counter % 100 == 0:
                    print(f"DEBUG: UI rendering frame {self.ui_counter}", flush=True)
                self.ui_counter += 1

                frame = self.detector.current_frame
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_resize = cv2.resize(img, (720, 540))
                img_pil = Image.fromarray(img_resize)
                img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(720, 540))
                
                self.cam_label.configure(image=img_tk, text="")
                self.cam_label._image_ref = img_tk # Keep persistent reference
                
                if self.detector.shape:
                    self.status_badge.configure(text="[ ANALYSIS LOCKED ]" if self.detector.locked else "[ SCANNING LIVE ]", 
                                               text_color="#2ecc71" if self.detector.locked else COLORS["accent"])
                    
                    self.shape_val.configure(text=self.detector.shape.upper())
                    self.conf_bar.set(self.detector.confidence / 100)
                    
                    self.sym_bar.set(self.detector.symmetry_score / 100)
                    self.sym_val.configure(text=f"{self.detector.symmetry_score}%")
                    self.sym_desc.configure(text=self.detector.symmetry_label)

                    # Update Skin UI
                    self.skin_val.configure(text=f"TONE: {self.detector.skin_tone.upper()}", text_color=self.detector.skin_color)
                    self.palette_val.configure(text=f"RANGE: {self.detector.skin_palette}")
                    
                    if self.detector.shape != self.last_shape:
                        self.refresh_recommendations()
                        self.last_shape = self.detector.shape
        
        self.after(20, self.update_gui)

    def refresh_recommendations(self):
        for widget in self.rec_container.winfo_children():
            widget.destroy()

        hair_list, beard_list = [], []
        if self.detector.gender == "male" and isinstance(self.detector.rec, tuple):
            hair_list = self.detector.rec[0].split(',')
            beard_list = self.detector.rec[1].split(',')
        else:
            hair_list = str(self.detector.rec).split(',')
            beard_list = []

        hair_header = ctk.CTkLabel(self.rec_container, text="Hair Styles:", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["secondary"])
        hair_header.pack(anchor="w", pady=(2, 0))
        for h in hair_list:
            ctk.CTkLabel(self.rec_container, text=f"• {h.strip()}", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10)

        if beard_list:
            beard_header = ctk.CTkLabel(self.rec_container, text="Beard Styles:", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["secondary"])
            beard_header.pack(anchor="w", pady=(5, 0))
            for b in beard_list:
                ctk.CTkLabel(self.rec_container, text=f"• {b.strip()}", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=10)

    def on_close(self):
        self.detector.stop()
        self.destroy()

if __name__ == "__main__":
    detector = FaceDetector()
    app = App(detector)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
