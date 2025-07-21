import cv2
import mediapipe as mp
import joblib
import math

# Pontos do corpo
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15


def calculate_elbow_angle(landmarks):
    a = landmarks[LEFT_SHOULDER]
    b = landmarks[LEFT_ELBOW]
    c = landmarks[LEFT_WRIST]

    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    dot = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab * mag_cb == 0:
        return 0
    angle = math.acos(dot / (mag_ab * mag_cb))
    return math.degrees(angle)


def main(video_path):
    # Carrega modelo treinado
    model = joblib.load('angle_quality_model.pkl')

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angle = calculate_elbow_angle(landmarks)

            # Prever usando o modelo: precisa passar array 2D
            pred = model.predict([[angle]])[0]

            label = "Bom" if pred == 1 else "Ruim"

            # Desenhar resultado
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Angle: {int(angle)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Classificacao: {label}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if pred == 1 else (0, 0, 255), 2)

        cv2.namedWindow("Analise de Push-up", cv2.WINDOW_NORMAL)

        cv2.imshow("Analise de Push-up", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python analyze_video.py data/sample_videos/pushup_video.mp4")
    else:
        main(sys.argv[1])
