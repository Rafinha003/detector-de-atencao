import cv2
from camera import Camera
from face_mesh import FaceMeshDetector
from eye_metrics import eye_aspect_ratio, LEFT_EYE, RIGHT_EYE
from attencion_logic import AttentionLogic
from head_pose import get_head_pose

def main():
    cam = Camera()
    detector = FaceMeshDetector()
    attention = AttentionLogic()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        h, w, _ = frame.shape
        results = detector.process(frame)

        status = "SEM ROSTO"
        percent_atento = 0
        percent_distraido = 0
        pitch, yaw = None, None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2

            pitch, yaw = get_head_pose(landmarks, w, h)


            status, percent_atento, percent_distraido = attention.update(ear, pitch, yaw)

        color = (0, 255, 0) if status == "ATENTO" else (0, 0, 255)

        cv2.putText(
            frame,
            f"Status: {status}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        cv2.putText(
            frame,
            f"ATENTO: {percent_atento:.0f}%  DISTRAIDO: {percent_distraido:.0f}%",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Attention Detector", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
