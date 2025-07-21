import sys
import cv2
import mediapipe as mp
from processing.pushup_counter import PushupCounter
import csv
import math
import os
from datetime import datetime

# Constantes de pontos do corpo
LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15


# Garante que a pasta exista
os.makedirs('data', exist_ok=True)

# Gera nome único baseado no horário
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"data/training_data_{timestamp}.csv"

# Abre novo arquivo CSV
csv_file = open(csv_path, 'w', newline='')

# Cria o writer e escreve o cabeçalho
writer = csv.writer(csv_file)
writer.writerow(['angle', 'label'])


def calculate_elbow_angle(landmarks):
    """Calcula o ângulo entre ombro, cotovelo e punho esquerdo"""
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


def main():
    if len(sys.argv) < 2:
        print("Uso: python main.py caminho/do/video.mp4")
        return

    video_path = sys.argv[1]
    counter = PushupCounter()

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

    cv2.namedWindow("Push-up Counter", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            # Desenha os pontos e conexões
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            # Atualiza contador
            count = counter.update(results.pose_landmarks)
            cv2.putText(frame, f"Pushups: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Calcula ângulo do cotovelo
            angle = calculate_elbow_angle(results.pose_landmarks.landmark)

            # Exibe ângulo na tela
            cv2.putText(frame, f"Elbow Angle: {int(angle)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Push-up Counter", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('b') and results.pose_landmarks:
            writer.writerow([angle, 1])  # Bom exemplo
            print(f"Salvo: Bom ({angle:.2f})")
        elif key == ord('r') and results.pose_landmarks:
            writer.writerow([angle, 0])  # Ruim
            print(f"Salvo: Ruim ({angle:.2f})")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    csv_file.close()


if __name__ == "__main__":
    main()
