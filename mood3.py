import cv2
from deepface import DeepFace
import pygame
import time

# Initialize Pygame mixer
pygame.mixer.init()

# Emotion to sound mapping
emotion_sounds = {
    "happy": "sounds/happy.mp3",
    "sad": "sounds/sad.mp3",
    "angry": "sounds/angry.mp3",
    "neutral": "sounds/neutral.mp3",
    "surprise": "sounds/surprise.mp3"
}

# Play corresponding sound
def play_sound_for_emotion(emotion):
    sound_file = emotion_sounds.get(emotion)
    if sound_file:
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        except Exception as e:
            print("Sound error:", e)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Improve frame quality
    frame = cv2.resize(frame, (640, 640))  # Upscale for better detection
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)  # Brighten image

    try:
        # Analyze emotion with enforced detection disabled
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        emotion_probs = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']

        print(f"Detected Emotion: {dominant_emotion} | Probabilities: {emotion_probs}")

        # Consider sadness if it is even slightly close to dominant
        if emotion_probs['sad'] > 20 and dominant_emotion != 'sad':
            dominant_emotion = 'sad'

        if dominant_emotion != last_emotion:
            play_sound_for_emotion(dominant_emotion)
            last_emotion = dominant_emotion

    except Exception as e:
        print("Detection error:", e)

    cv2.imshow("MoodMate - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
