import cv2
from time import sleep
import imutils
# 추출할 이미지 로드
# videocapture() 부분에 local 영상 경로 입력 (확장자포함.)
vidcap = cv2.VideoCapture('video/e.mp4') 
print(vidcap.isOpened()) # 영상이 제대로 들어왔는지 확인 True, false => 영상못불러온거임.

count = 1
success = True
num = 1
while success:
  success,image = vidcap.read()

  if count % 35 == 0:
    cv2.imwrite("cutf/joint/e%d.jpg" % num, image) #image: 벡터화된 ndarray 형태의 자료구조

    print("saved image %d.jpg" % num)
    num += 1
  if cv2.waitKey(10) == 27:       
      break
  count += 1


  