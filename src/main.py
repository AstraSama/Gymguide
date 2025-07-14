import cv2
from camera.capture import Camera
from processing.pose_estimator import PoseEstimator
from processing.pushup_counter import PushupCounter


def main():
    cam = Camera()
    estimator = PoseEstimator()
    counter = PushupCounter()

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        landmarks = estimator.estimate(frame)
        if landmarks:
            reps = counter.update(landmarks)
            cv2.putText(frame, f"Reps: {reps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Flexão Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
