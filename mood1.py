import cv2
from deepface import DeepFace
import pygame
import time

# Initialize Pygame mixer
pygame.mixer.init()

# Define mapping of emotions to sound files
emotion_sounds = {
    "happy": "sounds/happy.mp3",
    "sad": "sounds/sad.mp3",
    "angry": "sounds/angry.mp3",
    "neutral": "sounds/neutral.mp3",
    "surprise": "sounds/surprise.mp3"
}

def play_sound_for_emotion(emotion):
    sound_file = emotion_sounds.get(emotion)
    if sound_file:
        try:
            # Stop current sound before playing a new one
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
        except Exception as e:
            print("Sound error:", e)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        print("Detected Emotion:", emotion)

        if emotion != last_emotion:
            play_sound_for_emotion(emotion)
            last_emotion = emotion

    except Exception as e:
        print("Detection error:", e)

    cv2.imshow("MoodMate - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
