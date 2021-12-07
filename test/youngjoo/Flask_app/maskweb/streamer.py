import time
import cv2
import imutils
import platform
import numpy as np
from threading import Thread
from queue import Queue
import cvlib as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import requests,json,time
from datetime import datetime
import base64

class Streamer:

    def __init__(self):

        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
        print('[wandlab] ', 'OpenCL : ', cv2.ocl.haveOpenCL())

        self.capture = None
        self.thread = None
        self.width = 640
        self.height = 360
        self.stat = False
        self.current_time = time.time()
        self.preview_time = time.time()
        self.sec = 0
        self.Q = Queue(maxsize=128)
        self.started = False

    def run(self, src=1):

        self.stop()

        if platform.system() == 'Windows':
            self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        else:
            self.capture = cv2.VideoCapture(1)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.thread is None:
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = False
            self.thread.start()

        self.started = True

    def stop(self):

        self.started = False

        if self.capture is not None:
            self.capture.release()
            self.clear()

    def update(self):

        no_stable = 0
        yes_stable = 0
        model = load_model('mask_model_1027.h5')
        model.summary()

        while True:

            if self.started:
                (grabbed, frame) = self.capture.read()

                if grabbed:

                    face, confidence = cv.detect_face(frame)
                    # 사람 없을 때
                    if face == []:
                        # print("no people")
                        cv2.putText(frame, "no person...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    for idx, f in enumerate(face):

                        (startX, startY) = f[0], f[1]
                        (endX, endY) = f[2], f[3]

                        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
                            0] and 0 <= endY <= frame.shape[0]:

                            face_region = frame[startY:endY, startX:endX]

                            face_region1 = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)

                            x = img_to_array(face_region1)
                            x = np.expand_dims(x, axis=0)
                            x = preprocess_input(x)

                            prediction = model.predict(x)
                            
                            if prediction < 0.5:  # 마스크 미착용으로 판별되면,
                                
                                no_stable += 1
                                yes_stable = 0
                            
                                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                                Y = startY - 10 if startY - 10 > 10 else startY + 10
                                text = "No Mask ({:.2f}%) ".format((1 - prediction[0][0]) * 100)
                                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                print(no_stable, yes_stable)
                                if no_stable > 50 :
                                    #화면 캡쳐
                                    cap_time = datetime.today().strftime('%y%m%d%H%M%S')
                                    cv2.imwrite("cap/%s.jpg" % cap_time, frame) #image: 벡터화된 ndarray 형태의 자료구조
                                    cv2.putText(frame, "Fail Mask Detection", (startX, Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                    if no_stable > 55 :
                                        # django 백엔드 서버 ip
                                        upload = {'file':open("cap/%s.jpg" % cap_time, 'rb')}
                                        url = 'http://127.0.0.1:8000/acto/mask/off/'
                                        # d = {"base64img" : base64_img,
                                        #     "test" : 'testmsg'
                                        # }
                                        r = requests.post(url,files=upload)
                                        no_stable = 0
                                        if r.status_code == '200':
                                            print("--------------- success _-------------------")
                                    
                            elif prediction > 0.5:  # 마스크 착용으로 판별되면
                                no_stable = 0
                                yes_stable += 1
                                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                                Y = startY - 10 if startY - 10 > 10 else startY + 10
                                text = "Mask ({:.2f}%)".format(prediction[0][0] * 100)
                                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                print(no_stable, yes_stable)
                                if yes_stable > 60 :
                                    # django 백엔드 서버 ip
                                    cv2.putText(frame, "Success Mask Detection", (startX, Y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    if yes_stable > 65:
                                        yes_stable = 0
                                        url = 'http://127.0.0.1:8000/acto/mask/on/'
                                        data = ''
                                        r = requests.post(url)
                                    
                                    # if r.status_code == '200':
                                    #     print("--------------- success _-------------------")
                            else:
                                no_stable = 0
                                yes_stable = 0
                                ####### stable 판단 -> api 통신 -> 디벨롭 : time()함수로 바꾸기 #######
                                
                           # cv2.imshow("frame",frame)
                self.Q.put(frame)

    def clear(self):

        with self.Q.mutex:
            self.Q.queue.clear()

    def read(self):

        return self.Q.get()

    def blank(self):

        return np.ones(shape=[self.height, self.width, 3], dtype=np.uint8)

    def bytescode(self):

        if not self.capture.isOpened():

            frame = self.blank()

        else:

            frame = imutils.resize(self.read(), width=int(self.width))

            if self.stat:
                cv2.rectangle(frame, (0, 0), (120, 30), (0, 0, 0), -1)
                fps = 'FPS : ' + str(self.fps())
                cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        return cv2.imencode('.jpg', frame)[1].tobytes()

    def fps(self):

        self.current_time = time.time()
        self.sec = self.current_time - self.preview_time
        self.preview_time = self.current_time

        if self.sec > 0:
            fps = round(1 / (self.sec), 1)

        else:
            fps = 1

        return fps

    def __exit__(self):
        print('* streamer class exit')
        self.capture.release()