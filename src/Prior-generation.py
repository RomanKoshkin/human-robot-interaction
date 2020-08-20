#!/usr/bin/python2.7
import sys
sys.path.insert(0, '/home/torobo/catkin_ws/src/torobo_robot/torobo_rnn/scripts')
from torobo_rnn_utils import *
import os
import numpy as np
import rospy
import math
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
import ctypes
from utils2 import Utils
import utils2
from NRL import NRL
from collections import deque
import time
from torobo_operator import ToroboOperator




torobo = ToroboOperator()

# ROS Image message -> OpenCV2 image converter
#from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image

#print gravity_compensation_effort
time.sleep(2)
#bridge = CvBridge()

la_min = [70.0, 80.0, -30.0, 0.0]
la_max = [100.0, 90.0,  30.0, 15.0]
DATA_DIR_0='/home/torobo/catkin_ws/src/tutorial/src/data/Joints_9_right_arm/0_1_0_1_0_0_1_1/primitive_0_0.csv'
DATA_DIR_1='/home/torobo/catkin_ws/src/tutorial/src/data/Training_data_2/primitive_1_0.csv'
DATA_DIR_2='/home/torobo/catkin_ws/src/tutorial/src/data/Training_data_3/primitive_2_0.csv'
ra_min = la_min
ra_max = la_max


# Define the minimum and maximum of the head angles
head_min = [-20.0, -10.0]
head_max = [20.0,  20.0]

ARM_JNTS_FIXED = 0.0
TORSO_FIXED = 0.0

def to_radians(input_data):
    data=input_data
    for i in range(data.shape[0]):
    #pos_list=data[i]
        for j in range(len(data[i])):
            data[i][j]=math.radians(data[i][j])
    return data

def model_to_radians(input_data):
    for i in range(len(input_data)):
        input_data[i]=math.radians(input_data[i])
    return input_data

# The movement will be generated for 4 seconds
PLAY_DURATION = 10

# We'll capture 10 data points per second
CAPTURE_RATE = 10

#NUM_TRIALS = 5
# Set velocity override here (0-100[%])
#set_velocity_override(torobo, 50.0)
# Set softness override here (0-100[%])
set_softness_override(torobo, 50.0)
servo_on(torobo)


class Trial(object):
    def __init__(self):
        self.nrl=NRL()
        self.cwd=os.getcwd()
        self.ut=Utils(self.cwd)
        self.nZ = '4,1'
        self.nD = '40,10'             
        #nZ = '4,2,1'
        #nD = '40,20,10'
        prop_path = "/home/torobo/catkin_ws/src/tutorial/src/data/config/model_2_seq_2.d"
        self.nrl.newModel(prop_path.encode('ascii'))
        self.nrl.load()
        self.nDof = self.nrl.getNDof()
        self.stateBufferSize = self.nrl.getStateBufferSize(self.nD, self.nZ, self.ut.delimiter)
        if self.nDof > 0:
            winSize = 15
            winBufferSize = winSize * self.nDof
            winBuffdata = deque(maxlen=winSize) # circular buffer
            primId = 2

            # The e_w parameters set bellow assume the network has two layers
            # as in the original distribution of the sources
            # in case more layers are set by changing the properties.d file,
            # the same dimension for e_w must be considered
            e_w = [0.025,0.025]
            self.start=False

            expTimeSteps = 30
            postdiction_epochs = 15
            alpha = 0.1
            beta1 = 0.9
            beta2 = 0.999
            storeStates = False
            storeER = False
            showERLog = False
            self.nrl.e_enable(primId,\
                         winSize,
                         (ctypes.c_float * len(e_w))(*e_w),
                         expTimeSteps,
                         postdiction_epochs,
                         (ctypes.c_float)(alpha),
                         (ctypes.c_float)(beta1),
                         (ctypes.c_float)(beta2), 
                         storeStates, 
                         storeER)

    def generate_pos(self):
        self.tgt_pos_buffer = np.zeros((self.nDof,), dtype=float)
        dataOut = (ctypes.c_float * self.nDof)(*self.tgt_pos_buffer)
        self.stateBufferSize = self.nrl.getStateBufferSize(self.nD, self.nZ, self.ut.delimiter)
        self.nrl.e_generate(dataOut)
        self.tgt_pos = np.frombuffer(dataOut, np.float32)
        return self.tgt_pos

model=Trial()

servo_on(torobo)
time.sleep(1)

move_homepos(torobo)

torobo.set_control_mode(ToroboOperator.TORSO_HEAD, 'all', 'position')
torobo.set_control_mode(ToroboOperator.LEFT_ARM, 'all', 'position')
torobo.set_control_mode(ToroboOperator.RIGHT_ARM, 'all', 'position')
time.sleep(1)
move_startpos(torobo)

Demonstration=True
if Demonstration==True:
    print "Demonstrating Recorded Trajectory"
    print"PRIMITIVE_0"
    cor_saved=np.loadtxt(DATA_DIR_0, delimiter=",")
    cor_saved_1=np.loadtxt(DATA_DIR_1, delimiter=",")
    cor_saved_2=np.loadtxt(DATA_DIR_2, delimiter="," )
    for i in xrange(cor_saved.shape[0]):
        #left_cor=list(cor_saved[i][:6])
        
        right_cor=list(array_radians(cor_saved[i]))
        #torobo.move(ToroboOperator.LEFT_ARM, positions=left_cor)
        #time.sleep(1)
        torobo.move(ToroboOperator.RIGHT_ARM, positions=right_cor)
        time.sleep(1)
    print 'PRIMITIVE_1'
    move_startpos(torobo)
    for i in xrange(cor_saved_1.shape[0]):
        #
        #
       
        #left_cor=list(cor_saved_1[i][:6])
        right_cor=list(array_radians(cor_saved_1[i]))
        #torobo.move(ToroboOperator.LEFT_ARM, positions=left_cor)
        torobo.move(ToroboOperator.RIGHT_ARM, positions=right_cor)
        time.sleep(1)
    print "PRIMITIVE_2"
    move_startpos(torobo)
    for i in xrange(cor_saved_2.shape[0]):
        #left_cor=list(cor_saved_2[i][:6])
        #right_cor=list(cor_saved_2[i][6:12])
       # torobo.move(ToroboOperator.LEFT_ARM, positions=left_cor)
       
        right_cor=list(array_radians(cor_saved_2[i]))
        torobo.move(ToroboOperator.RIGHT_ARM, positions=right_cor)
        time.sleep(1)

move_startpos(torobo)
time.sleep(1)
servo_off(torobo)
PRIOR_GENERATION=False

if PRIOR_GENERATION==True:
    print "Demontrating prior generation"
    servo_on(torobo)
    time.sleep(1)
    model_list=[]
    model_data=np.array(model.generate_pos())
    for i in xrange(300):
        full_cor=np.array(model.generate_pos())
        #model_data=np.vstack((model_data, full_cor))
        right_cor=list(array_radians(full_cor)) 
        #left_cor=full_cor[:6]
        #print left_cor
        #model_list.append(left_cor[0])
        #right_cor=full_cor[6:]
        #right_cor=full_cor
        #torobo.move(ToroboOperator.LEFT_ARM, positions=left_cor)
        torobo.move(ToroboOperator.RIGHT_ARM, positions=right_cor)
        time.sleep(1)
        #time.sleep(1)
    #np.savetxt('/home/shpurov/catkin_ws/src/reaching_task/scripts/data/model_data.csv', model_data, delimiter=",")
    move_startpos(torobo)
    time.sleep(1)
    servo_off(torobo)
