import cv2
import mediapipe as mp

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Abrir o vídeo
video = cv2.VideoCapture("test_video.mp4")  # substitua pelo nome real do arquivo

# Verificar se o vídeo foi carregado corretamente
if not video.isOpened():
    raise FileNotFoundError("❌ Não foi possível abrir o vídeo.")

# Criar a janela e ajustar o tamanho para o vídeo
cv2.namedWindow("Pose Estimada", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Estimada", int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Inicializar o MediaPipe Pose
with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
    while True:
        ret, frame = video.read()
        if not ret:
            break  # fim do vídeo

        # Converter o frame de BGR para RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar o frame
        results = pose.process(frame_rgb)

        # Desenhar os landmarks, se detectados
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Exibir o frame com os landmarks desenhados
        cv2.imshow("Pose Estimada", frame)

        # Espera 1ms e verifica se a tecla 'q' foi pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera o vídeo e fecha as janelas
video.release()
cv2.destroyAllWindows()
