# Face Detection System

A real-time Face Detection System built with Python and OpenCV. It uses Haar Cascade face detection, supports webcam input, and includes a Streamlit app that detects, counts, crops and saves multiple faces.

## Features
- Detect faces in images
- Real-time detection through webcam
- Bounding boxes around faces
- Face counting, cropping and saving
- Streamlit web app with an adjustable strictness slider

## Tech Used
Python, OpenCV, NumPy, Streamlit

## How to Run

    pip install -r requirements.txt
    python detect.py
    python webcam.py
    python -m streamlit run app.py

Put your own photo in a folder called `sample_images` and change the file name inside `detect.py`.
