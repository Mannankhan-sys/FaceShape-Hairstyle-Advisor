import cv2
import mediapipe as mp
from face_shape import classify_face_shape
from recommendations import get_recommendations

def get_gender():
    while True:
        gender = input("Enter gender (male/female): ").strip().lower()
        if gender in ["male", "female"]:
            return gender
        print("Invalid input. Please type 'male' or 'female'.")

def main():
    # 1️⃣ Ask gender first
    gender = get_gender()

    # 2️⃣ Initialize MediaPipe
    mp_face = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)  # small points

    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            h, w, _ = frame.shape

            for face in results.multi_face_landmarks:
                # Draw smaller landmarks
                mp_draw.draw_landmarks(frame, face, mp_face.FACEMESH_TESSELATION,
                                       landmark_drawing_spec=mp_style,
                                       connection_drawing_spec=mp_style)

                # Collect landmarks
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

                # Detect face shape
                shape = classify_face_shape(landmarks)

                # Smooth prediction
                history.append(shape)
                if len(history) > 10:
                    history.pop(0)
                shape = max(set(history), key=history.count)

                # Get recommendations
                rec = get_recommendations(shape, gender)

                # --- Display on screen ---
                y0 = 40
                dy = 30
                cv2.putText(frame, f"Face Shape: {shape}", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display hairstyle & beard list
                if gender == "male":
                    cv2.putText(frame, f"Hair Styles:", (20, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    for i, style in enumerate(rec[0].split(',')):
                        cv2.putText(frame, f"- {style.strip()}", (40, y0+(i+2)*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
                    cv2.putText(frame, f"Beard Styles:", (20, y0+(len(rec[0].split(','))+2)*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    for i, style in enumerate(rec[1].split(',')):
                        cv2.putText(frame, f"- {style.strip()}", (40, y0+(len(rec[0].split(','))+3+i)*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                else:
                    cv2.putText(frame, f"Hair Styles:", (20, y0+dy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                    for i, style in enumerate(rec.split(',')):
                        cv2.putText(frame, f"- {style.strip()}", (40, y0+(i+2)*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)

        # Show the frame
        cv2.imshow("Face Shape Detector", frame)

        # --- Exit Options ---
        # Press 'q' or ESC to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
