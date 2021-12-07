import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('wash_hand_model.h5')
model.summary()

# open webcam
cap = cv.VideoCapture(0)


if not cap.isOpened():
    print("Could not open webcam")
    exit()

while cap.isOpened():
    ret, img = cap.read()

    if ret:
        cv.imshow('frame_color', img)  #

        hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # 손 색상 범위 설정
        lower = np.array([0, 40, 30], dtype="uint8")
        upper = np.array([50, 255, 255], dtype="uint8")

        skinRegionHSV = cv.inRange(hsvim, lower, upper)
        blurred = cv.blur(skinRegionHSV, (2, 2))
        ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)

        # 인식한 부분을 원본 이미지 변경하는 부분
        img_result = cv.bitwise_and(img, img, mask=thresh)

        resize_img = cv.resize(img_result, (150, 150), interpolation=cv.INTER_AREA)

        x = img_to_array(resize_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        prediction = model.predict(x)
        print(prediction)
        stage=[]
        wash_stage=max(prediction[0])
        for i in range(4):
            if wash_stage == prediction[0][i]:
                stage.append(i)
        print(stage)
        cv.putText(img, str(stage[0]), (0, 0), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.imshow("thresh", thresh)
        cv.imshow("img", img)
        cv.imshow("img_result", img_result)

    # press "Q" to stop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
cap.release()
cv.destroyAllWindows()