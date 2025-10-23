import cv2
from deepface import DeepFace
import pygame
import time

emotion_colors = {
    "happy":(0,255,255),
    "sad":(255,0,0),
    "angry":(0,0,255),
    "surprise":(255,0,255),
    "neutral":(0,255,0)
}

# Initialize Pygame mixer
pygame.mixer.init()

# Emotion to sound mapping
emotion_sounds = {
    "happy": "sounds/happy.mp3",
    "sad": "sounds/sad.mp3",
    "angry": "sounds/angry.mp3",
    # "neutral": "sounds/neutral.mp3",
    "surprise": "sounds/surprise.mp3",
    "fear": "sounds/horror.mp3"
}

# Function to play sound for detected emotion
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

def draw_corner_glow(frame,color,thickness=60,alpha=0.4):
    h,w,_=frame.shape
    overlay=frame.copy()

    cv2.rectangle(overlay,(0,0),(thickness,thickness),color,-1)
    cv2.rectangle(overlay,(w-thickness,0),(w,thickness),color,-1)
    cv2.rectangle(overlay,(0,h-thickness),(thickness,h),color,-1)
    cv2.rectangle(overlay,(w-thickness,h-thickness),(w,h),color,-1)
    cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

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

    # Enhance frame quality for better detection
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
    # frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        emotion_probs = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']
        cv2.putText(frame,dominant_emotion,(150,100),fontFace=1,fontScale=2,color=(122,100,210),thickness=2)
        

        # print(f"Detected: {dominant_emotion} | Probs: {emotion_probs}")

        # Manual override if fear or sad have high probability
        # if emotion_probs.get('sad', 0) > 25 and dominant_emotion != 'sad':
        #     dominant_emotion = 'sad'
        # elif emotion_probs.get('fear', 0) > 12 and dominant_emotion != 'fear':
        #     dominant_emotion = 'fear'
        # elif emotion_probs.get('angry', 0) > 18 and dominant_emotion != 'angry':
        #     dominant_emotion = 'angry'
       
        if dominant_emotion != last_emotion:
            play_sound_for_emotion(dominant_emotion)
            last_emotion = dominant_emotion
        color=emotion_colors.get(dominant_emotion,(255,255,255))
        draw_corner_glow(frame,color)


    except Exception as e:
        print("Detection error:", e)

    cv2.imshow("MoodMate - Press Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
