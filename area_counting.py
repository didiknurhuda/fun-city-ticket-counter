
import cv2

def print_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")

if __name__ == "__main__":
    #cap = cv2.VideoCapture(r"D:\\UNINDRA\\Penelitian\\Computer Vision\\Deteksi Tiket Fun City\\20250615_173820.mp4")
    cap = cv2.VideoCapture(1)  # Uncomment this line to use webcam instead of video file
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("frame", frame)
        cv2.setMouseCallback("frame", print_coordinates)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
