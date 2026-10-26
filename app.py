import os
import cv2
import numpy as np
import streamlit as st

st.title("Face Detection System")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

min_neighbors = st.slider("Strictness (minNeighbors)", 1, 10, 5)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    height, width = image.shape[:2]
    scale = 600 / height
    image = cv2.resize(image, (int(width * scale), 600))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    # Keep a clean copy so the crops don't contain the blue box lines
    original = image.copy()

    crops = []
    for (x, y, w, h) in faces:
        crops.append(original[y:y + h, x:x + w])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    st.write("Faces found:", len(faces))

    if len(crops) > 0:
        st.subheader("Cropped faces")

        columns = st.columns(5)
        for i, face in enumerate(crops):
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            columns[i % 5].image(face_rgb, caption=f"Face {i + 1}")

        if st.button("Save faces to folder"):
            os.makedirs("detected_faces", exist_ok=True)
            for i, face in enumerate(crops, start=1):
                cv2.imwrite(f"detected_faces/face_{i}.jpg", face)
            st.success(f"Saved {len(crops)} faces in the detected_faces folder")