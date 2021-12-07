import cv2 as cv
import numpy as np
import os

# 학습하고자 하는 파일명 넣는 곳
cap = cv.VideoCapture("test.mp4")

width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.avi', fourcc, 30.0, (int(width), int(height)))

while (True):
    ret, img = cap.read()  # Read 결과와 frame

    if (ret):
        cv.imshow('frame_color', img)  # 컬러 화면 출력

        hsvim = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        # 손 색상 범위 설정
        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")

        skinRegionHSV = cv.inRange(hsvim, lower, upper)
        blurred = cv.blur(skinRegionHSV, (2, 2))
        ret, thresh = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY)

        #인식한 부분을 원본 이미지 변경하는 부분
        img_result = cv.bitwise_and(img, img, mask=thresh)

        cv.imshow("thresh", thresh)

        cv.imshow("img_result", img_result)

        out.write(img_result)
        if cv.waitKey(20) == ord('q'):
            break

cap.release()
cv.destroyAllWindows()