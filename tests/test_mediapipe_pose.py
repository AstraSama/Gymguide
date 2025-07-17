import cv2
import mediapipe as mp

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Carregar a imagem
imagem = cv2.imread('image.png')

# Verificação para garantir que a imagem foi carregada corretamente
if imagem is None:
    raise FileNotFoundError("❌ A imagem 'model_test.png' não foi encontrada no diretório atual.")

# Converter a imagem de BGR para RGB
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Processar a imagem com MediaPipe Pose
with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
    resultado = pose.process(imagem_rgb)

    # Se detectar landmarks, desenhar
    if resultado.pose_landmarks:
        mp_drawing.draw_landmarks(
            imagem,
            resultado.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

# Exibir imagem com pose desenhada
cv2.imshow("Resultado da Pose", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
