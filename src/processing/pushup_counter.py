import math

# Índices dos pontos do corpo conforme modelo BlazePose (usado no MediaPipe)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16


class PushupCounter:
    def __init__(self, threshold_down=0.6, threshold_up=0.4):
        self.state = 'down'
        self.count = 0
        self.th_down = threshold_down
        self.th_up = threshold_up

    def _get_body_angle(self, a, b, c):
        """Calcula o ângulo entre os vetores AB e CB"""
        va = (a.x - b.x, a.y - b.y)
        vb = (c.x - b.x, c.y - b.y)
        dot = va[0] * vb[0] + va[1] * vb[1]
        mag = math.hypot(*va) * math.hypot(*vb)
        if mag == 0:
            return 0
        return math.degrees(math.acos(dot / mag))

    def update(self, landmarks):
        """
        landmarks: NormalizedLandmarkList da pose principal.
        """

        if not landmarks or len(landmarks.landmark) <= RIGHT_WRIST:
            return self.count  # evita erro caso não tenha todos os pontos

        angle = self._get_body_angle(
            landmarks.landmark[LEFT_SHOULDER],
            landmarks.landmark[LEFT_ELBOW],
            landmarks.landmark[LEFT_WRIST],
        )

        # Normalização baseada no ângulo de referência (90 graus como base)
        norm = (angle - 90) / 90

        # Lógica de contagem
        if norm > self.th_down and self.state == 'up':
            self.state = 'down'
        elif norm < self.th_up and self.state == 'down':
            self.state = 'up'
            self.count += 1

        return self.count
