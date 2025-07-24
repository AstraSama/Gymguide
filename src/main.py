import sys
import cv2
import mediapipe as mp
from processing.pushup_counter import PushupCounter
import csv
import math
import os
import numpy as np

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15


def calculate_elbow_angle(landmarks):
    if len(landmarks) <= LEFT_WRIST:
        return 0
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
        print("Uso: python3 src/main.py caminho/do/video.mp4 [modo: lateral|frontal]")
        return

    video_path = sys.argv[1]
    mode = 'lateral'  # padrão
    if len(sys.argv) >= 3:
        mode_arg = sys.argv[2].lower()
        if mode_arg in ['lateral', 'frontal']:
            mode = mode_arg
        else:
            print(f"Modo '{mode_arg}' inválido, usando padrão 'lateral'.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    angle_list = []
    counter = PushupCounter(mode=mode)

    cv2.namedWindow("Push-up Collector", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angle = calculate_elbow_angle(landmarks)
            angle_list.append(angle)

            # Atualiza repetições
            count = counter.update(results.pose_landmarks)

            # Desenha pontos e conexões
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Angle: {int(angle)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Reps: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {mode}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Push-up Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in [ord('b'), ord('r')]:
            label = 1 if key == ord('b') else 0
            if angle_list:
                np_angles = np.array(angle_list)
                row = [
                    np.mean(np_angles),
                    np.std(np_angles),
                    np.min(np_angles),
                    np.max(np_angles),
                    np.max(np_angles) - np.min(np_angles),  # amplitude
                    counter.count,
                    label
                ]

                os.makedirs('data', exist_ok=True)
                csv_name = "training_data_stats.csv"
                csv_exists = os.path.exists(f"data/{csv_name}")

                with open(f"data/{csv_name}", 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not csv_exists:
                        writer.writerow(['mean', 'std', 'min', 'max', 'amplitude', 'reps', 'label'])
                    writer.writerow(row)

                print(f"Salvo! Label: {'Bom' if label else 'Ruim'} — {row}")

                # Reinicia para próxima coleta
                angle_list = []
                counter.count = 0
                counter.state = 'down'  # reset estado do contador

            else:
                print("Nenhum ângulo coletado!")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
