import cv2
import time

def capture_photo():
    time.sleep(2)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite('captured_photo.jpg', frame)
        print("Photo captured and saved!")
    else:
        print("Failed to capture photo")

    cap.release()

if __name__ == "__main__":
    capture_photo()
