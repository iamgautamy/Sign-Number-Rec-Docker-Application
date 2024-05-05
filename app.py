from flask import Flask, render_template, Response,flash,Blueprint
import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
import pickle
app = Flask(__name__,template_folder="../template")

def image_processed(hand_img):
    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        # print(data)
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return clean
    except:
        return np.zeros([1, 63], dtype=int)[0]

# Load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

def gen_frames():  
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            data = image_processed(frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 100)
            fontScale = 2
            color = (255, 0, 0)
            thickness = 5
            if np.sum(data) == 0:
                frame = cv2.putText(frame, 'No Hand Detected', org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
            else:
                data = np.array(data)
                y_pred = svm.predict(data.reshape(-1, 63))
                frame = cv2.putText(frame, f'Prediction: {y_pred[0]}', org, font,
                                    fontScale, color, thickness, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()


@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80,debug=True)
    app.logger.error('An error occurred')