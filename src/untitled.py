from __future__ import print_function
import sys, os
import time
import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, '/home/torobo/catkin_ws/src/tutorial/PyTorch-YOLOv3')
sys.path.insert(0, '/home/torobo/catkin_ws/src/torobo_robot/torobo_rnn/scripts')

from detect_upd import Recog

# from torobo_rnn_utils__upd import *
from torobo_rnn_utils__upd3 import *

import rospy
import numpy as np
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
import cv2

from adjustExtForce_v3Controller import ExtForce

bridge = CvBridge()
torobo = ToroboOperator()

recog = Recog()


class CV2im(threading.Thread):
  
  def __init__(self):
    threading.Thread.__init__(self)
    self.imagebuff = 0
    time.sleep(1)
    self.keepgoing = False
    rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
    time.sleep(1)
 
  def image_callback(self, msg):
    self.imagebuff = bridge.imgmsg_to_cv2(msg, "rgb8")

  def stop(self):
    self.keepgoing = False
   
  def run(self):
    self.keepgoing = True
    while True and self.keepgoing:
      detections = recog.detect(self.imagebuff)
      if detections is not None:
        for det in detections:
          for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            image = cv2.rectangle(self.imagebuff, (x1,y1), (x2,y2), (255, 0, 0), 3)
        cv2.imshow('asdfasdfs', cv2.cvtColor(self.imagebuff, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(3) & 0XFF
        if k== 27:
          break
    cv2.destroyAllWindows()

def main():
  cv2im = CV2im()
  cv2im.start()

if __name__ == '__main__':
  main()
