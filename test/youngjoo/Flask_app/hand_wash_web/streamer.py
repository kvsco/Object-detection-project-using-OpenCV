
import cv2
import imutils
import platform
import numpy as np
from threading import Thread
from queue import Queue
import requests

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import time
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils
# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands


def getHandType(image, results, draw=True, display=True):
    # Create a copy of the input image to write hand type label on.
    output_image = image.copy()

    # Initialize a dictionary to store the classification info of both hands.
    hands_status = {'Right': False, 'Left': False, 'Right_index': None, 'Left_index': None}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand.
        hand_type = hand_info.classification[0].label

        # Update the status of the found hand.
        hands_status[hand_type] = True

        # Update the index of the found hand.
        hands_status[hand_type + '_index'] = hand_index

        # Check if the hand type label is specified to be written.
        if draw:
            # Write the hand type on the output image.
            cv2.putText(output_image, hand_type + ' Hand Detected', (10, (hand_index + 1) * 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and the hands status dictionary that contains classification info.
        return output_image, hands_status


def detectHandsLandmarks(image, hands, display=True):
    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # # Check if landmarks are found.
    # if results.multi_hand_landmarks:
    #     pass
    #     # Iterate over the found hands.
    #     # for hand_landmarks in results.multi_hand_landmarks:
    #     #     # Draw the hand landmarks on the copy of the input image.
    #     #     mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
    #     #                               connections=mp_hands.HAND_CONNECTIONS)

            # Check if the original input image and the output image are specified to be displayed.
    # if display:
    #
    #     # Display the original input image and the output image.
    #     # plt.figure(figsize=[15, 15])
    #     # plt.subplot(121);
    #     # plt.imshow(image[:, :, ::-1]);
    #     plt.title("Original Image");
    #     plt.axis('off');
    #     plt.subplot(122);
    #     plt.imshow(output_image[:, :, ::-1]);
    #     plt.title("Output");
    #     plt.axis('off');
    #
    # # Otherwise
    # else:

        # Return the output image and results of hands landmarks detection.
    return output_image, results


def detectHandsLandmarks(image, hands, display=True):
    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found.
    if results.multi_hand_landmarks:
        pass
        # Iterate over the found hands.
        # for hand_landmarks in results.multi_hand_landmarks:
        # Draw the hand landmarks on the copy of the input image.
        #     mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
        # Check if the original input image and the output image are specified to be displayed.
    if display:
        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121);
        plt.imshow(image[:, :, ::-1]);
        plt.title("Original Image");
        plt.axis('off');
        plt.subplot(122);
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and results of hands landmarks detection.
        return output_image, results


def drawBoundingBoxes(image, results, hand_status, padd_amount=10, draw=True, display=True):
    global x1_r, y1_r, x2_r, y2_r, x1_l, y1_l, x2_l, y2_l
    # Create a copy of the input image to draw bounding boxes on and write hands types labels.
    output_image = image.copy()

    # Initialize a dictionary to store both (left and right) hands landmarks as different elements.
    output_landmarks = {}

    # Get the height and width of the input image.
    height, width, _ = image.shape
    two_hand = [0, 0]
    # Iterate over the found hands.
    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

        # Initialize a list to store the detected landmarks of the hand.
        landmarks = []

        # Iterate over the detected landmarks of the hand.
        for landmark in hand_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

        # Get all the x-coordinate values from the found landmarks of the hand.
        x_coordinates = np.array(landmarks)[:, 0]

        # Get all the y-coordinate values from the found landmarks of the hand.
        y_coordinates = np.array(landmarks)[:, 1]


        # Check if the hand we are iterating upon is the right one.
        if hand_status['Right_index'] == hand_index:

            x1_r = int(np.min(x_coordinates) - padd_amount)
            y1_r = int(np.min(y_coordinates) - padd_amount)
            x2_r = int(np.max(x_coordinates) + padd_amount)
            y2_r = int(np.max(y_coordinates) + padd_amount)


            # Update the label and store the landmarks of the hand in the dictionary.
            label = 'Right Hand'
            output_landmarks['Right'] = landmarks
            two_hand[0] = 1
        # Check if the hand we are iterating upon is the left one.
        elif hand_status['Left_index'] == hand_index:

            x1_l = int(np.min(x_coordinates) - padd_amount)

            y1_l = int(np.min(y_coordinates) - padd_amount)
            if x1_l <= 0 and y1_l <=0:
                x1_l = 0
                y1_l =0
            elif x1_l <= 0:
                x1_l =0
            elif y1_l<=0:
                y1_l=0
            x2_l = int(np.max(x_coordinates) + padd_amount)
            y2_l = int(np.max(y_coordinates) + padd_amount)

            # Update the label and store the landmarks of the hand in the dictionary.
            label = 'Left Hand'
            output_landmarks['Left'] = landmarks
            two_hand[1] = 1

        # Check if the bounding box and the classified label is specified to be written.
        if draw:
            # Draw the bounding box around the hand on the output image.
            # cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.LINE_8)
            pass
            # Write the classified label of the hand below the bounding box drawn.
            # cv2.putText(output_image, label, (x1, y2 + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (20, 255, 155), 1,cv2.LINE_AA)
    #
    # 손 바운딩 박스 그리기
    if two_hand[0] == 1 and two_hand[1] == 1:

        if y1_r <= y1_l:
            two_hand_bbox_y1 = y1_r
        else:
            two_hand_bbox_y1 = y1_l
        if two_hand_bbox_y1 <=0:
            two_hand_bbox_y1 = 0

        if y2_r >= y2_l:
            two_hand_bbox_y2 = y2_r
        else:
            two_hand_bbox_y2 = y2_l

        if x1_l <0 :
            two_hand_bbox_x1 =0
        else:
            two_hand_bbox_x1 = x1_l
        two_hand_bbox_x2 = x2_r

        if two_hand_bbox_x1> two_hand_bbox_x2:
            two_hand_bbox_x1 ,two_hand_bbox_x2= two_hand_bbox_x2, two_hand_bbox_x1


        # cv2.rectangle(output_image, (two_hand_bbox_x1, two_hand_bbox_y1), (two_hand_bbox_x2, two_hand_bbox_y2),(0, 255, 0), 3, cv2.LINE_8)
        hand_box = [[two_hand_bbox_x1, two_hand_bbox_y1], [two_hand_bbox_x2, two_hand_bbox_y2]]
    elif two_hand[0] == 1:
        one_hand_xr = x1_r - int((x2_r - x1_r)*0.5)
        if one_hand_xr <=0:
            one_hand_xr = 0
        if y1_r<=0:
            y1_r =0
        # cv2.rectangle(output_image, (one_hand_xr, y1_r), (x2_r, y2_r),(0, 255, 0), 3, cv2.LINE_8)
        hand_box = [[one_hand_xr, y1_r], [x2_r, y2_r]]
    elif two_hand[1] == 1:
        one_hand_xl = x2_l + int((x2_l - x1_l)*0.5)
        if y1_l<=0:
            y1_l=0
        # cv2.rectangle(output_image, (x1_l, y1_l), (one_hand_xl, y2_l),(0, 255, 0), 3, cv2.LINE_8)
        hand_box = [[x1_l, y1_l], [one_hand_xl, y2_l]]
    else:
        hand_box=[]
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1]);
        plt.title("Output Image");
        plt.axis('off');

    # Otherwise
    else:

        # Return the output image and the landmarks dictionary.
        return output_image, output_landmarks, hand_box





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
        self.sec = 1
        self.Q = Queue(maxsize=128)
        self.started = False

    def run(self, src=0):

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
        hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                                     min_detection_confidence=0.7, min_tracking_confidence=0.4)
        model1 = load_model('./stage1.h5')
        model2 = load_model('./stage2.h5')
        model3_l = load_model('./stage3_r.h5')
        model3_r = load_model('./stage3_l.h5')
        model4_r = load_model('./stage4_r.h5')
        model4_l = load_model('./stage4_l.h5')

        stage_progress = [0] * 5
        stage_list = []
        stage_count_list = [0] * 5
        not_detect_hand_time = []

        while True:

            if self.started:
                (grabbed, frame) = self.capture.read()
                frame = cv2.flip(frame, 1)

                if grabbed:
                    frame = cv2.flip(frame, 1)
                    frame, results = detectHandsLandmarks(frame, hands_video, display=False)
                    # if stage_progress[1] > 5:
                    #     cv2.putText(frame, "stage1 clear", (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # else:
                    #     cv2.putText(frame, "stage1 " + str(stage_progress[1]), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    #                 (0, 0, 255), 2)
                    # if stage_progress[2] > 5:
                    #     cv2.putText(frame, "stage2 clear", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # else:
                    #     cv2.putText(frame, "stage2 " + str(stage_progress[2]), (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    #                 (0, 0, 255), 2)
                    # if stage_progress[3] > 5:
                    #     cv2.putText(frame, "stage3 clear", (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # else:
                    #     cv2.putText(frame, "stage3 " + str(stage_progress[3]), (75, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    #                 (0, 0, 255), 2)

                    # if stage_progress[4] > 5:
                    #     cv2.putText(frame, "stage4 clear", (75, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # else:
                    #     cv2.putText(frame, "stage4 " + str(stage_progress[4]), (75, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    #                 (0, 0, 255), 2)
                    # 인식되는 손이 3초 이상 지속 되었을 때 기록 초기화
                    if not results.multi_hand_landmarks:

                        not_detect_hand = round(time.time())
                        if len(not_detect_hand_time) == 0:
                            not_detect_hand_time.append(not_detect_hand)

                        elif len(not_detect_hand_time) != 0 and not_detect_hand_time[-1] != not_detect_hand:
                            not_detect_hand_time.append(not_detect_hand)
                            # 5초 이상 지속되면 리스트 삭제
                            if not_detect_hand_time[-1] - not_detect_hand_time[0] >= 4:
                                not_detect_hand_time = []
                                stage_progress = [0] * 5
                                url = "http://127.0.0.1:8000/acto/wash/"
                                h_number = "0"
                                data = {}
                                data['hand_number'] = h_number
                                r = requests.post(url,data=data)
                                if r.status_code == 200:
                                    print("-------------success--------------")
                                else:
                                    print("44444444444444444", r.status_code)
                                

                    print("stage_progress", stage_progress)
                    print("not_detect_hand_list", not_detect_hand_time)
                    if results.multi_hand_landmarks:
                        # 손 인식 되었을 때 초기화
                        not_detect_hand_time = []

                        # Perform hand(s) type (left or right) classification.
                        _, hands_status = getHandType(frame.copy(), results, draw=False, display=False)

                        # Draw bounding boxes around the detected hands and write their classified types near them.
                        frame, _, hand_box = drawBoundingBoxes(frame, results, hands_status, display=False)
                        # print("hand_box",hand_box)
                        (startX, startY) = int(hand_box[0][0] * 0.85), int(hand_box[0][1] * 0.8)
                        (endX, endY) = int(hand_box[1][0] * 1.1), int(hand_box[1][1] * 1.1)

                        hand_in_img = frame[startY:endY, startX:endX, :]
                        cv2.imshow("hand_in_img", hand_in_img)

                        resize_img = cv2.resize(hand_in_img, (224, 224), interpolation=cv2.INTER_AREA)

                        x = img_to_array(resize_img)
                        x = np.expand_dims(x, axis=0)
                        x = preprocess_input(x)

                        prediction1 = model1.predict(x)
                        prediction2 = model2.predict(x)
                        # # 3단계 중 큰 값 사용
                        prediction3_r = model3_r.predict(x)
                        prediction3_l = model3_l.predict(x)
                        prediction3 = max(prediction3_r, prediction3_l)
                        # # 4단계 중 큰 값 사용
                        prediction4_r = model4_r.predict(x)
                        prediction4_l = model4_l.predict(x)
                        prediction4 = max(prediction4_r, prediction4_l)
                        #
                        prediction_list = [prediction1, prediction2,prediction3,prediction4]

                        predict = max(prediction_list)
                        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        # print("prediction1", prediction1)
                        # print("prediction2", prediction2)
                        # print("prediction3", prediction3)
                        # print("prediction3_r", prediction3_r)
                        # print("prediction3_l", prediction3_l)
                        # print("prediction4_r", prediction4_r)
                        # print("prediction4_l", prediction4_l)
                        # # 예측이 0.6 이하이면 버리고 이상이면 리스트에 저장

                        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 3, cv2.LINE_8)
                        if predict < 0.5:
                            # print("not wash hand")
                            cv2.putText(frame, "not wash hand", (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 0, 255), 2)
                        else:
                            stage = prediction_list.index(predict) + 1
                            # print("stage", stage)
                            stage_list.append(stage)
                            cv2.putText(frame, "stage" + str(stage), (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 0, 0), 2)

                        if len(stage_list) == 10:
                            for i in range(1, 5):
                                stage_count_list[i] = stage_list.count(i)
                                max_count = max(stage_count_list)
                            for i in range(1, 5):
                                if max_count == stage_count_list[i]:
                                    stage_progress[i] += 1
                                    print("-------------i------------", i, "type i------", type(i))
                                    if i == 1 :
                                        url = "http://127.0.0.1:8000/acto/wash/"
                                        h_number = "1"
                                        data = {}
                                        data['hand_number'] = h_number
                                        r = requests.post(url,data=data)
                                        if r.status_code == 200:
                                            print("-------------success--------------")
                                        else:
                                            print("44444444444444444", r.status_code)
                                    elif i == 2:
                                        url = "http://127.0.0.1:8000/acto/wash/"
                                        h_number = "2"
                                        data = {}
                                        data['hand_number'] = h_number
                                        r = requests.post(url,data=data)
                                        if r.status_code == 200:
                                            print("-------------success--------------")
                                        else:
                                            print("44444444444444444", r.status_code)
                                    elif i == 3:
                                        url = "http://127.0.0.1:8000/acto/wash/"
                                        h_number = "3"
                                        data = {}
                                        data['hand_number'] = h_number
                                        r = requests.post(url,data=data)
                                        if r.status_code == 200:
                                            print("-------------success--------------")
                                        else:
                                            print("44444444444444444", r.status_code)
                                    elif i == 4:
                                        url = "http://127.0.0.1:8000/acto/wash/"
                                        h_number = "4"
                                        data = {}
                                        data['hand_number'] = h_number
                                        r = requests.post(url,data=data)
                                        if r.status_code == 200:
                                            print("-------------success--------------")
                                        else:
                                            print("44444444444444444", r.status_code)
                                    
                            stage_list = []

                        # print("frame_time",frame_time)
                    cv2.imshow("img", frame)
                    # cv2.imshow("img_result", img_result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
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