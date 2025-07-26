import mediapipe as mp
import os
import sys
import json
import pandas as pd

# Adiciona o diretório 'src' ao sys.path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# flake8: noqa: E402
import cv2
import joblib
import math
import numpy as np

from processing.pushup_counter import PushupCounter
from llm_feedback import gerar_feedback  # <-- novo

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
    print(f"🎬 Iniciando análise do vídeo: {video_path}")
    counter = PushupCounter(mode='lateral')  # ou 'lateral', ou 'frontal'
    model = joblib.load('pushup_quality_model.pkl')
    angles = []
    predicted_label = None

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

    cv2.namedWindow("Análise de Push-up", cv2.WINDOW_NORMAL)

    feature_cols = ['mean', 'std', 'min', 'max', 'amplitude', 'reps']
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Falha ao ler frame. Encerrando loop.")
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            angle = calculate_elbow_angle(landmarks)
            angles.append(angle)

            reps = counter.update(results.pose_landmarks)

            # if predicted_label is None and reps >= 3 and len(angles) >= 5:
            #     features = {
            #         'mean': np.mean(angles),
            #         'std': np.std(angles),
            #         'min': np.min(angles),
            #         'max': np.max(angles),
            #         'amplitude': np.max(angles) - np.min(angles),
            #         'reps': reps
            #     }

            #     X_input = pd.DataFrame([[
            #         features['mean'], features['std'], features['min'],
            #         features['max'], features['amplitude'], features['reps']
            #     ]], columns=feature_cols)
            #     prediction = model.predict(X_input)[0]

            #     predicted_label = "Bom" if prediction == 1 else "Ruim"

            #     # Salvar dados e enviar ao LLM
            #     dados = {
            #         "video_path": video_path,
            #         "mode": counter.mode,  # novo!
            #         **features,
            #         "classificacao": predicted_label,
            #         "angle_list": angles[-30:]
            #     }


            #     with open("data/last_analysis.json", "w") as f:
            #         json.dump(dados, f, indent=2)

            #     print("⚙️ Chamando gerar_feedback()...")
            #     feedback = gerar_feedback(dados)
            #     print("\n💬 Feedback gerado pelo agente:\n")
            #     print(feedback)
            #     print("\n" + "="*60 + "\n")

            # Desenhos
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"Angle: {int(angle)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Repetições: {reps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 255, 255), 2)

            if predicted_label:
                cv2.putText(frame, f"Classificação: {predicted_label}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if predicted_label == "Bom" else (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Analisando...", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Análise de Push-up", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("🎞️ Total de frames no vídeo:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    # ⚠️ Gera feedback mesmo que reps < 3
    print("⚠️ Vídeo terminou. Verificando se é possível gerar feedback com dados disponíveis...")

    if predicted_label is None and len(angles) >= 5:
        print("✅ Entrou no bloco de feedback final.")
        features = {
            'mean': np.mean(angles),
            'std': np.std(angles),
            'min': np.min(angles),
            'max': np.max(angles),
            'amplitude': np.max(angles) - np.min(angles),
            'reps': counter.count
        }
        print(f"📊 Features calculadas: {features}")


        X_input = pd.DataFrame([[
                features['mean'], features['std'], features['min'],
                features['max'], features['amplitude'], features['reps']
            ]], columns=feature_cols)
        prediction = model.predict(X_input)[0]
        predicted_label = "Bom" if prediction == 1 else "Ruim"
        print(f"🧠 Predição feita: {predicted_label}")

        dados = {
            "video_path": video_path,
            "mode": counter.mode,
            **features,
            "classificacao": predicted_label,
            "angle_list": angles[-30:]
        }

        with open("data/last_analysis.json", "w") as f:
            json.dump(dados, f, indent=2)
        print("💾 Arquivo last_analysis.json salvo com sucesso.")

        feedback = gerar_feedback(dados)
        # feedback = "🔁 Simulação de feedback: tudo rodando até aqui!"
        print("\n💬 Feedback gerado pelo agente:\n")
        print(feedback)
        print("\n" + "=" * 60 + "\n")
    else:
        print("❗ Não houve dados suficientes para gerar feedback.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analyze_video.py caminho/do/video.mp4")
    else:
        main(sys.argv[1])