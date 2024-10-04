import cv2
import dlib
from scipy.spatial import distance
import serial
import time

# Khởi tạo cổng serial
# s = serial.Serial('/dev/tty.usbmodem1401', 9600)
# Hàm tính toán tỷ lệ khía cạnh của mắt (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Khởi tạo video capture và các detector
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("/Users/hiep/Downloads/wordspacd/python/Driver-s-Drowsiness-Detection-using-OpenCV-Python-main/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")

# Khởi tạo biến để theo dõi thời gian bắt đầu và số khung hình
start_time = None
no_eye_start_time = None
CONSECUTIVE_FRAMES = 20
frame_count = 0
count_sleep = 0

while True:
    # Lấy một khung hình
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    if len(faces) > 0:
        # Chọn khuôn mặt gần nhất dựa trên diện tích lớn nhất
        largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        face_landmarks = dlib_facelandmark(gray, largest_face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = eye_aspect_ratio(leftEye)
        right_ear = eye_aspect_ratio(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        if EAR < 0.26:
            if start_time is None:
                start_time = time.time()
            frame_count += 1
            elapsed_time = time.time() - start_time

            if elapsed_time >= 1.5:
                # s.write(b'a')
                count_sleep += 1
        
                cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
                cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("Drowsy")
        else:
            start_time = None
            frame_count = 0
            # s.write(b'c')
        # Đặt lại bộ đếm thời gian không phát hiện mắt
        no_eye_start_time = None
        print(EAR)
    else:
        if no_eye_start_time is None:
            no_eye_start_time = time.time()
        elapsed_time = time.time() - no_eye_start_time
        if elapsed_time >= 2:
            # s.write(b'a')
            no_eye_start_time = None
            print("No eye detected for 2 seconds")
    cv2.imshow("Are you Sleepy", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
