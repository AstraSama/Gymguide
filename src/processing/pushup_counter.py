import math

# Índices dos pontos do corpo conforme BlazePose
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23  # para o modo frontal


class PushupCounter:
    def __init__(self, mode='lateral', threshold_down=0.6, threshold_up=0.4):
        self.state = 'down'
        self.count = 0
        self.mode = mode
        self.th_down = threshold_down
        self.th_up = threshold_up
        self.initial_shoulder_y = None

    def _calculate_elbow_angle(self, shoulder, elbow, wrist):
        """Calcula o ângulo entre ombro, cotovelo e punho"""
        a = (shoulder.x - elbow.x, shoulder.y - elbow.y)
        b = (wrist.x - elbow.x, wrist.y - elbow.y)

        dot = a[0]*b[0] + a[1]*b[1]
        mag_a = math.hypot(*a)
        mag_b = math.hypot(*b)

        if mag_a * mag_b == 0:
            return 0

        angle = math.acos(dot / (mag_a * mag_b))
        return math.degrees(angle)

    def update(self, landmarks):
        if not landmarks or len(landmarks.landmark) <= RIGHT_WRIST:
            return self.count

        if self.mode == 'lateral':
            angle = self._calculate_elbow_angle(
                landmarks.landmark[LEFT_SHOULDER],
                landmarks.landmark[LEFT_ELBOW],
                landmarks.landmark[LEFT_WRIST],
            )
            norm = (angle - 90) / 90
            if norm > self.th_down and self.state == 'up':
                self.state = 'down'
            elif norm < self.th_up and self.state == 'down':
                self.state = 'up'
                self.count += 1

        elif self.mode == 'frontal':
            shoulder_y = landmarks.landmark[LEFT_SHOULDER].y
            hip_y = landmarks.landmark[LEFT_HIP].y

            delta = abs(shoulder_y - hip_y)

            if self.initial_shoulder_y is None:
                self.initial_shoulder_y = shoulder_y

            # Ajuste os thresholds conforme necessário
            if delta > 0.15 and self.state == 'up':
                self.state = 'down'
            elif delta < 0.07 and self.state == 'down':
                self.state = 'up'
                self.count += 1

        return self.count
