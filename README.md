# Facial Recognition and Authentication 👤🔐

This is a Python program that uses OpenCV and face recognition libraries to perform real-time facial recognition and authentication. The program captures video from the webcam, detects faces, and matches them against a pre-trained dataset to identify individuals. It calculates the accuracy of the match and displays the results.

## Features 🌟

- Real-time facial recognition using webcam input.
- Compares detected faces against pre-trained encodings.
- Calculates accuracy based on facial distance and match threshold.
- Displays the matched name and accuracy for each recognized face.
- Returns the most matched name after a certain number of frames.

## Installation and Usage 🛠️

1. Clone this repository to your local machine.
2. Make sure you have Python and the required libraries installed.
3. Run the `recognize_face.py` script using the command: `python recognize_face.py`.
4. The program will capture video from your webcam and display the results in the console.

## Dependencies 📦

- `cv2` (OpenCV): For capturing and processing video frames.
- `face_recognition`: For face detection and encoding.
- `numpy`: For array operations.

## Configuration ⚙️

- You can adjust the `faceMatchThreshold` and `accuracy` thresholds in the `getAccuracy` function to fine-tune matching and accuracy.
- Update the `face_recognition_data` directory path to point to your own dataset of face images.

## Acknowledgments 🙌

- The `face_recognition` library simplifies face recognition tasks.
- Inspired by the idea of using facial recognition for authentication.

## Note 📝

- This program is designed for educational purposes and might require additional improvements for production-level security.

## Author 🧑‍💻

[Arshit Arora]


