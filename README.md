# MoodMate
An OpenCV-based mood detection tool that listens to your face and responds with matching sounds. MoodMate detects facial expressions via your webcam and plays short audio clips that correspond to the detected mood — a playful assistant to help demonstrate basic emotion recognition and multimedia feedback.

Features
Real-time face and expression detection using OpenCV.
Maps detected moods/emotions to audio clips and plays them automatically.
Lightweight, easy to extend (swap models or audio files).
Designed as an educational/demo project to explore CV + audio feedback.

The application will:

Open the webcam
Detect faces and compute emotion probabilities
Select the most likely emotion
Play the corresponding audio clip
Press q (or the configured key) to quit the app.

How it works (high level)
Capture video frames from the webcam using OpenCV.
Detect face(s) in the frame (Haar cascades, DNN, or other detector).
Crop/normalize the face image and feed it into an emotion classifier (simple heuristic or a trained model).
Map the predicted emotion to an audio file and play it back to the user.
This project is intentionally modular so you can replace the face detector, the emotion model, or the audio playback mechanism.