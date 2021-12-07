# -*- encoding: utf-8 -*-
#-------------------------------------------------#
# Date created          : 2020. 8. 18.
# Date last modified    : 2020. 8. 19.
# Author                : chamadams@gmail.com
# Site                  : http://wandlab.com
# License               : GNU General Public License(GPL) 2.0
# Version               : 0.1.0
# Python Version        : 3.6+
#-------------------------------------------------#

import time
import cv2
import imutils
import platform
import numpy as np
from threading import Thread
from queue import Queue

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/road.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_boolean('dis_cv2_window', False, 'disable cv2 window during the process') # this is good for the .ipynb

flag_model = 'yolov4'
flag_framework = 'tf'
flag_iou= 0.45
flag_score = 0.25
flag_tiny = True
flag_weights = './checkpoints/yolov4-416'
flag_dis_cv2_window= False


class Streamer :
    
    def __init__(self ):
        
        if cv2.ocl.haveOpenCL() :
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
        
    def run(self, src = 0 ) :
        
        self.stop()

        if platform.system() == 'Windows' :        
            self.capture = cv2.VideoCapture( 1 , cv2.CAP_DSHOW )
        
        else :
            self.capture = cv2.VideoCapture( 1 )
            
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if self.thread is None :
            self.thread = Thread(target=self.update, args=())
            self.thread.daemon = False
            self.thread.start()
        
        self.started = True
    
    def stop(self):
        
        self.started = False
        
        if self.capture is not None :
            
            self.capture.release()
            self.clear()
            
    def update(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = 416
        # video_path = FLAGS.video

        print("input_size", input_size)
        # print("Video from: ", video_path)
        vid = cv2.VideoCapture(1)
        
        if flag_framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=flag_weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        else:
            saved_model_loaded = tf.saved_model.load(flag_weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        # if FLAGS.output:
        #     # by default VideoCapture returns float instead of int
        #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     fps = int(vid.get(cv2.CAP_PROP_FPS))
        #     codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        #     out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

        frame_id = 0
        while True:

            if self.started :
                grabbed, frame = self.capture.read()
                frame=cv2.flip(frame, 1)
                if grabbed:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame)
                else:
                    if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                        print("Video processing complete")
                        break
                    raise ValueError("No image! Try with another video format")

                frame_size = frame.shape[:2]
                image_data = cv2.resize(frame, (input_size, input_size))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)
                prev_time = time.time()

                if flag_framework == 'tflite':
                    interpreter.set_tensor(input_details[0]['index'], image_data)
                    interpreter.invoke()
                    pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                    if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                        boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                        input_shape=tf.constant([input_size, input_size]))
                    else:
                        boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                        input_shape=tf.constant([input_size, input_size]))
                else:
                    batch_data = tf.constant(image_data)
                    pred_bbox = infer(batch_data)
                    for key, value in pred_bbox.items():
                        boxes = value[:, :, 0:4]
                        pred_conf = value[:, :, 4:]

                boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                    scores=tf.reshape(
                        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                    max_output_size_per_class=50,
                    max_total_size=50,
                    iou_threshold=flag_iou,
                    score_threshold=flag_score
                )
                pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

                image = utils.draw_bbox(frame, pred_bbox)
                curr_time = time.time()
                exec_time = curr_time - prev_time
                result = np.asarray(image)
                info = "time: %.2f ms" % (1000 * exec_time)
                #print(info)

                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # frame=cv2.flip(frame, 1)

                #이거 아래는 없어도 됩니다
                # if not flag_dis_cv2_window:
                #     cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)

                #     cv2.imshow("result", frame)
                #     if cv2.waitKey(1) & 0xFF == ord('q'): break

                # if FLAGS.output:
                #     out.write(result)

                frame_id += 1

                if grabbed :
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

        else :
            
            frame = imutils.resize(self.read(), width=int(self.width) )
        
            if self.stat :  
                cv2.rectangle( frame, (0,0), (120,30), (0,0,0), -1)
                fps = 'FPS : ' + str(self.fps())
                cv2.putText  ( frame, fps, (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1, cv2.LINE_AA)
            
            
        return cv2.imencode('.jpg', frame )[1].tobytes()
    
    def fps(self):
        
        self.current_time = time.time()
        self.sec = self.current_time - self.preview_time
        self.preview_time = self.current_time
        
        if self.sec > 0 :
            fps = round(1/(self.sec),1)
            
        else :
            fps = 1
            
        return fps
    def __exit__(self) :
        print( '* streamer class exit')
        self.capture.release()