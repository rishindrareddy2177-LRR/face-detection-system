import cv2

image = cv2.imread("sample_images/manyfaces.jpg")

height, width = image.shape[:2]
scale = 600 / height
image = cv2.resize(image, (int(width * scale), 600))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=6,
    minSize=(30, 30)
)

print("Faces found:", len(faces))

for i, (x, y, w, h) in enumerate(faces, start=1):
    print(i, "x =", x, "y =", y, "width =", w)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(image, str(i), (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
