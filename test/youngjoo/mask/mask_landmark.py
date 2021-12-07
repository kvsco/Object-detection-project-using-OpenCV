import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
import dlib

import time


ALL = list(range(0, 68))
RIGHT_EYEBROW = list(range(17, 22))
LEFT_EYEBROW = list(range(22, 27))
RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAWLINE = list(range(0, 17))

index = ALL

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('model.h5')
# model.summary()
no_stable = 0
yes_stable = 0

# open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()
ok_mask = []
no_mask = []
# loop through frames
pre_time = time.time()

while webcam.isOpened():

    # read frame from webcam
    status, frame = webcam.read()
    # cv2.imshow("mask nomask frame", frame)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dets = detector(img_gray, 1)

    for face in dets:
        print("face",face)
        shape = predictor(frame, face)  # 얼굴에서 68개 점 찾기
        print(shape)

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i, pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv2.circle(frame, pt_pos, 2, (0, 255, 0), -1)

        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()),
                     (0, 0, 255), 3)


    if not status:
        print("Could not read frame")

        exit()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # print("face",face)
    # print("confidence",confidence)

    if face == []:
        print("no people")
        cv2.putText(frame, "no person...", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # loop through detected faces
    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]


        # faceBoxRectangleS = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)
        # print("faceBoxRectangleS", faceBoxRectangleS)

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[
            0] and 0 <= endY <= frame.shape[0]:

            face_region = frame[startY:endY, startX:endX]

            # img_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

            # shape = predictor(frame, faceBoxRectangleS)
            # list_points = []
            # for p in shape.parts():
            #     list_points.append([p.x, p.y])
            # list_points = np.array(list_points)
            #
            # for i, pt in enumerate(list_points[index]):
            #     pt_pos = (pt[0], pt[1])
            #     cv2.circle(frame, pt_pos, 2, (0, 255, 0), -1)
            # cv2.rectangle(frame, (startX, startY), (endX, endY),
            #              (0, 0, 255), 3)

            # cv2.imshow("img_gray",img_gray)
            face_region1 = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)
            # cv2.imshow("mask nomask face_region1", face_region1)


            x = img_to_array(face_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            prediction = model.predict(x)
            print(prediction)

            if prediction < 0.5:  # 마스크 미착용으로 판별되면,
                no_mask.append("N")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "No Mask ({:.2f}%)".format((1 - prediction[0][0]) * 100)
                # cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "no mask!", (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:  # 마스크 착용으로 판별되면
                ok_mask.append("Y")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Mask ({:.2f}%)".format(prediction[0][0] * 100)
                cv2.putText(frame, text, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)



    # display output
    cv2.imshow("mask nomask classify", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()