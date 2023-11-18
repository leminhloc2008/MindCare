import cv2
from ultralytics import YOLO
import time
import math
import numpy as np
import imutils
import telepot
from deepface import DeepFace
import serial

from datetime import datetime
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from scipy.spatial.distance import pdist, squareform, cdist
from roboflow import Roboflow
from emotion_recognition import EmotionRecognizer

import pyaudio
import os
import wave
from sys import byteorder
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

from matplotlib import pyplot as plt
import pandas as pd
import sounddevice as sd

from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input

from utils import get_best_estimators

rf = Roboflow(api_key="sDRFXcrut0gyZAsXjhgu")
project = rf.workspace().project("abnormal-activities-u130g")
model = project.version(1).model

# Ghi âm
THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 30

def is_silent(snd_data):
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}


# Kết nối đến arduino
#ser = serial.Serial('COM3', 9600)


token = '6591943273:AAGp1NNT_GK3RgJ9E81XXP1zpUAaDYQREA4'
receiver_id = 5606318609

bot = telepot.Bot(token)

# Load DEEPFACE model
faceModel = DeepFace.build_model('Emotion')

# Cảm xúc
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

image = None

def start_detect(cap):
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    features = ["mfcc", "chroma", "mel"]
    detector = EmotionRecognizer(estimator_dict["BaggingClassifier"],
                                 emotions=["neutral", "calm", "happy", "fear", "disgust", "ps", "angry", "neutral",
                                           "sad"], features=features, verbose=0)
    detector.train()
    filename = "test.wav"
    FT = {}
    headCenterx = {}
    headCentery = {}
    u = {}
    v = {}
    writer = None

    isTurnedOn = False

    SOLID_BACK_COLOR = (41, 41, 41)

    num_mouse_points = 0
    frame_num = 0
    time_count = 0
    status = "Bình Thường"

    emotion_translations = {
        "angry": "Giận Dữ",
        "disgust": "Khó Chịu",
        "fear": "Sợ Hãi",
        "happy": "Hạnh Phúc",
        "neutral": "Trung Lập",
        "sad": "Buồn",
        "surprise": "Ngạc Nhiên",
    }

    abnormal_activity_translations = {
        "hang": "Treo Cổ",
        "Knife_Deploy": "Sử dụng dao",
        "Knife_Weapon": "Sử dụng dao",
        "Stabbing": "Sử dụng dao",
        "gun": "Sử dụng súng",
    }

    activity_translations = {
        "": "Khong xac dinh",
        "Drinking": "Uong Nuoc",
        "Fall_down": "Nga",
        "Lying_down": "Dang Nam",
        "Nearly_fall": "Sap Nga",
        "Walking": "Di bo",
        "Standing": "Dung",
        "Walking_on_Stairs": "Di cau thang",
        "Sitting": "Ngoi"
    }

    while True:
        frame_num += 1
        time_count += 1
        ret,frame = cap.read()
        frame = imutils.resize(frame, width=700)
        height, width, channels = frame.shape
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        start_time = time.time()

        if not ret:
            break

        record_to_file(filename)

        # Hiển thị
        frame_ = frame

        # Gửi tin nhắn

        # Đổi kích thước khung hình
        resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)

        # Đổi màu khung hình
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Xử lý ảnh cho DEEPFACE
        img = gray_frame.astype('float32') / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        # Nhận diện cảm xúc bằng Deep Face
        preds = faceModel.predict(img)
        emotion_idx = np.argmax(preds)
        emotion = emotion_labels[emotion_idx]

        if time_count == 15:
            bot.sendMessage(receiver_id, 'Đây là tin nhắn tự động, được thiết lập để gửi sau ' + str(time_count) + ' giây')
            bot.sendMessage(receiver_id, 'Cảm xúc hiện tại: ' + str(emotion_translations[emotion]))
            filename = "D:\\Documents\\ElderlyHealthCare\\photo\\sendImage.jpg"
            cv2.imwrite(filename, frame_)
            bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
            os.remove(filename)
            time_count = 0

        preds = model.predict(frame, confidence=20, overlap=30).json()
        detections = preds['predictions']
        for box in detections:
            x1 = box['x'] - box['width'] / 2
            x2 = box['x'] + box['width'] / 2
            y1 = box['y'] - box['height'] / 2
            y2 = box['y'] + box['height'] / 2
            classes = box['class']
            print(classes)
            if classes == "hang":
                cv2.rectangle(frame_, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame_, "Treo Co", (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print("FPS :", fps)

        cv2.putText(frame_, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)

        # Hiển thị
        cv2.imshow("MindCare", frame_)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Lưu video
        if writer is None:
            date_time = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
            OUTPUT_PATH = 'output\output {}.avi'.format(date_time)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (frame_.shape[1], frame_.shape[0]), True)
        writer.write(frame_)

        result = detector.predict(filename)
        print(result)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    writer.release()

def webcam_detect():
    cap = cv2.VideoCapture(0)
    start_detect(cap)

def start_video(video_path):
    cap = cv2.VideoCapture(video_path)
    start_detect(cap)

def streamCam(streamUrl):
    cap = cv2.VideoCapture(streamUrl)
    start_detect(cap)

def passCam(username, password):
    cap = cv2.VideoCapture("rtsp://" + username + ':' + password + "@192.168.1.64/1")
    start_detect(cap)

if __name__ == '__main__':
    webcam = False
    video_play = True
    #ser.write(b'L')
    # H: bật đèn, L: tắt đèn
    if webcam:
        webcam_detect()
    if video_play:
        start_video('test\hang.mp4')

    cv2.destroyWindow()