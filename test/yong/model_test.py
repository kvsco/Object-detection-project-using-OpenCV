import cvlib as cv
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image
 
model = load_model('vgg16_handwash_joint.h5')
model.summary()

categories = ["step_1", "step_2", "step_3", "step_4"] 

def Dataization(img_path):
    image_w = 150
    image_h = 150
    img = cv2.imread(img_path)
    print(img)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0]) 
    return (img/256) 
  
src = [] 
name = [] 
test = [] 
image_dir = "testhand/" #테스트할 이미지 폴더 경로

for file in os.listdir(image_dir): 
    if (file.find('.jpg') is not -1):       
        src.append(image_dir + file) 
        name.append(file) 
        test.append(Dataization(image_dir + file))

test = np.array(test) 
y = model.predict(test)
print(y)

print( len(y))
l_max = [0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(len(y)):
    max =0
    for j in y[i]:
        num = float(j)
        if num > max :
            max = num
    l_max[i] = max
print(l_max)

# if max == float(y[0][0]):
#     print("1번 step")
# elif max == float(y[0][1]):
#     print("2번 step") 
# elif max == float(y[0][2]):
#     print("3번 step") 
# elif max == float(y[0][3]):
#     print("4번 step") 