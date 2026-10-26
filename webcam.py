import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Step 8: open the webcam
cap = cv2.VideoCapture(0)

# Step 9: process frames one by one
while True:
    success, frame = cap.read()

    if not success:
        print("Could not read from the camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) == 27:
        break

# Step 10: release the camera
cap.release()
cv2.destroyAllWindows()