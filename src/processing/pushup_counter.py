import math


class PushupCounter:
    def __init__(self, threshold_down=0.6, threshold_up=0.4):
        self.state = 'down'  # esperando subida
        self.count = 0
        self.th_down = threshold_down
        self.th_up = threshold_up

    def _get_body_angle(self, a, b, c):
        # Calcula ângulo no ponto b entre a→b e c→b
        va = (a.x - b.x, a.y - b.y)
        vb = (c.x - b.x, c.y - b.y)
        dot = va[0]*vb[0] + va[1]*vb[1]
        mag = math.hypot(*va) * math.hypot(*vb)
        return math.degrees(math.acos(dot/mag))

    def update(self, landmarks):
        # Pega pontos de ombro, cotovelo e punho
        L = landmarks.landmark
        angle = self._get_body_angle(
            L[self.mp_idx('LEFT_SHOULDER')],
            L[self.mp_idx('LEFT_ELBOW')],
            L[self.mp_idx('LEFT_WRIST')]
        )
        # Normaliza entre 0 e 1
        norm = (angle - 90) / 90

        # Estado máquina simples
        if norm > self.th_down and self.state == 'up':
            self.state = 'down'
        if norm < self.th_up and self.state == 'down':
            self.state = 'up'
            self.count += 1
        return self.count

    @staticmethod
    def mp_idx(name):
        from mediapipe.solutions.pose import PoseLandmark
        return PoseLandmark[name].value
