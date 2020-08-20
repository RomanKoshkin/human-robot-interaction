#!/usr/bin/env python2.7

import os
import sys
import time
import copy
import pickle
import rospy
sys.path.insert(0, '/home/torobo/catkin_ws/src/torobo_robot/torobo_rnn/scripts')

# from torobo_rnn_utils import *

sys.path.insert(0, '/home/torobo/catkin_ws/src/tutorial/PyTorch-YOLOv3')
from torobo_rnn_utils__upd3 import *
from detect_upd import Recog, Recog2
from scipy import signal
import scipy
from cv_bridge import CvBridge, CvBridgeError
import cv2



import rospy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from NRL import NRL
from inspect import getmembers, isfunction
import ctypes

from torobo_rnn_utils__upd3 import HiddenPrints
from torobo_driver import torobo_easy_command

from termcolor import colored
from enum import Enum

def echo(txt):
    sys.stdout.write('\r {}'.format(txt))
    sys.stdout.flush()


class Experiment(object):
    
    def __init__(self):
        
        self.nrl = NRL()
        self.cwd="/home/torobo/catkin_ws/src/tutorial/PVRNN/scripts"

        # Parameters         
        self.simulation = False
        self.simInteration = False                  # ???????????????????????????????????????????????????????????????????????????????
        self.moveSimInteration = False              # whether the robot posture shows the bottom up signal ??????????????????????????
        self.postdiction_enable = True        
            
        self.kr = 1

        ## RNN parameters
        self.topDownId = 0        
        bottomUpId = 0        
        bottomUpDataPath = '/home/torobo/catkin_ws/src/tutorial/PVRNN/scripts/data/Training_data_3/primitive_{}_0.csv'.format(bottomUpId)      
        self.bottomUpData = []
        
        self.actJointMask = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]        
      
        self.nActDof = 12
        self.w = [1.0e-5, 1.0e-5]
        self.windowSize = 20        
        self.postdiction_epochs = 15             
        self.nBuffdata = self.windowSize * self.nActDof

        # experiment time
        self.T = 1000 
        
        # Adam optimization params
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.alpha = 0.1

        subject = 'test_1'

        self.expPath = self.cwd +"/data/experiment/{}".format(subject)
                
        if self.postdiction_enable:
            try:                        
                if (not os.path.exists(self.expPath)):
                    os.mkdir(self.expPath)                                    
            except OSError:
                self.sucess = False        
                print ("Creation of the directory %s failed" % self.expPath)
            else:
                print ("Successfully created the directory %s " % self.expPath)
                
        self.step = 0
        self.opt_elbo_list = []
        self.state_list = []
        self.signal_list  = []
        self.cur_pos_list  = []
        self.tgt_pos_list  = []        
        self.store_states_backend = False
        self.postdiction_store_backend = False
        self.stdoutPostiction = False
       
       
        
     
        prop_path="/home/torobo/catkin_ws/src/tutorial/src/data/config/modxp2.d"
        self.nrl.newModel(prop_path.encode('ascii'))   
        self.nrl.load()
        nZ = '4,1'
        nD = '40,10'             

        print("Experiment Parameters: ")
        print("######################################################")
       
        print("Num experiment steps:      {}".format(self.T))            
        print("ADAM alpha:                {}".format(self.alpha))
        print("ADAM beta1:                {}".format(self.beta1))
        print("ADAM beta2:                {}".format(self.beta2))            
        print("Enable postdiction:        {}".format(self.postdiction_enable))   
        print("Num postdiction epochs:    {}".format(self.postdiction_epochs))
        print("store states in backend:   {}".format(self.store_states_backend))            
        print("store postdiction backend: {}".format(self.postdiction_store_backend))            
        print("w:                         {}".format(self.w))
        print("Num d units:               {}".format(nD))
        print("Num z units:               {}".format(nZ))
        print("######################################################")

        self.stateBufferSize = self.nrl.getStateBufferSize(nD, nZ, ",")          

        self.nrl.e_enable(self.topDownId,
                            self.windowSize,
                            (ctypes.c_float * len(self.w))(*self.w),
                            self.T, 
                            self.postdiction_epochs,
                            (ctypes.c_float)(self.alpha),
                            (ctypes.c_float)(self.beta1),
                            (ctypes.c_float)(self.beta2), 
                            self.store_states_backend, 
                            self.postdiction_store_backend)      
                
        self.ERStartTime = self.windowSize
        self.npoints = (self.windowSize*self.nActDof)                

    def filterMask(self, _dat):
        filtered = []
        i = 0
        for m in self.actJointMask:
            if m:
                filtered.append(_dat[i])
            i = i + 1
        return filtered

    
class Tracker(threading.Thread):
    """ - LOADS THE YOLO MODEL
        - FORWARD PASSES THE FRAME THROUGH THE MODEL
        - STORES BOUNDING BOX COORDINATES IN SELF.
    """
    
    def __init__(self):
        threading.Thread.__init__(self)
        
        checkpoint = 42
        self.recog = Recog2(checkpoint)
        self.box_cener = (0.0, 0.0)
        self.x1, self.y1, self.x2, self.y2 = 0.0, 0.0, 0.0, 0.0
        self.new_hor = 0
        self.new_ver = 0
        self.Norm = scipy.stats.norm(0, 20)
        self.scale = self.Norm.pdf(0)
        self.lasttime = 0.0
        
        self.keepgoing = False

    def stop(self):
        self.keepgoing = False
        print(colored('TRACKER STOPPED', 'white', 'on_red'))

    def get_gaze_offset(self, cx, cy):
        return 640/2 - cx, 480/2 - cy
    
    def get_gaze_grad(self, offset_x, offset_y):
        ex = 0.2 * np.tanh(0.025*(offset_x))/ (1+self.Norm.pdf(offset_x)/self.scale*4)
        ey = 0.2 * np.tanh(0.025*(offset_y))/ (1+self.Norm.pdf(offset_x)/self.scale*4)
        return ex, ey

    def get_box_center(self, x1, y1, box_w, box_h):
        self.box_cener = (x1 + box_w/2, y1 + box_h/2)
        return x1 + box_w/2, y1 + box_h/2

    def track_obj(self, x,y,w,h):
        global torobo
        cx, cy = self.get_box_center(x, y, w, h)
        gaze_offset = self.get_gaze_offset(cx,cy)
        ex, ey = self.get_gaze_grad(*gaze_offset)
        hor, ver = get_cur_joints(torobo)[0][14:16]
        self.new_hor, self.new_ver = np.radians(hor)+ex, np.radians(ver)-ey
        self.move()
        
    def move(self):
        global torobo
        if time.time() - self.lasttime > 0.2:
            self.lasttime = time.time()
            hor, ver = self.new_hor, self.new_ver
            torobo.move_joint_my (controller_id=ToroboOperator.TORSO_HEAD,
                                  joint_ids = [0, 1, 2, 3],
                                  positions=[0, np.radians(38.0), hor, ver],
                                  velocities=None,
                                  accelerations=None,
                                  duration=0.2)
        else:
            pass       

    def run(self):
        global imagebuff
        self.keepgoing = True
        while True and self.keepgoing:
            detections = self.recog.detect(imagebuff)
            if detections is not None:
                for det in detections:
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        if cls_pred == 0:
                            box_w = x2 - x1
                            box_h = y2 - y1
                            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
                            x, y, w, h = x1, y1, box_w, box_h
                            self.track_obj(x, y, w, h)

    

class ExtForce(object):
    # SETS THE KINEMATIC MODEL PARAMETERS (BOTH IN HENDRY'S AND FACTORY FIRMWARE)

    class Mode(Enum):
        Teaching = 0
        Experiment = 1

    def runCommands(self, _commandList, _controller):
          for command in _commandList:
                torobo_easy_command.SendEasyCommandText(_controller, command)
                rospy.sleep(0.01)

    def registerParameters(self, tau_th, kp, sum_e_max, d, kr, theta_th):
            # left and right arm 
            for i in range(0,6):
                
                # constructing the commands for the left arm
                commandList = []                
                commandList.append("param " + str(i+1) + " fftauth " +       str(tau_th[i]))
                commandList.append("param " + str(i+1) + " ffkp " +          str(kp[i]))
                commandList.append("param " + str(i+1) + " ffsigmaemax " +   str(sum_e_max[i]))
                commandList.append("param " + str(i+1) + " ffdamping " +     str(d[i]))
                commandList.append("param " + str(i+1) + " ffkr " +          str(kr[i]))
                commandList.append("param " + str(i+1) + " softki " +        str(theta_th[i]))  


                # send to execution
                self.runCommands(commandList, "left_arm_controller")

            # right arm 
            for i in range(0,6):
                ii = i + 6 
                # constructing the commands for the right arm
                commandList = []
                commandList.append("param " + str(i+1) + " fftauth " +       str(tau_th[ii]))
                commandList.append("param " + str(i+1) + " ffkp " +          str(kp[ii]))
                commandList.append("param " + str(i+1) + " ffsigmaemax " +   str(sum_e_max[ii]))
                commandList.append("param " + str(i+1) + " ffdamping " +     str(d[ii]))
                commandList.append("param " + str(i+1) + " ffkr " +          str(kr[ii]))
                commandList.append("param " + str(i+1) + " softki " +        str(theta_th[ii])) 

                # send to execution
                self.runCommands(commandList, "right_arm_controller")
                            
            for i in range(0,4):
                ii = i + 12 
                # constructing the commands for the head-torso chain
                commandList = []                
                commandList.append("param " + str(i+1) + " fftauth " +       str(tau_th[ii]))
                commandList.append("param " + str(i+1) + " ffkp " +          str(kp[ii]))
                commandList.append("param " + str(i+1) + " ffsigmaemax " +   str(sum_e_max[ii]))
                commandList.append("param " + str(i+1) + " ffdamping " +     str(d[ii]))
                commandList.append("param " + str(i+1) + " ffkr " +          str(kr[ii]))
                commandList.append("param " + str(i+1) + " softki " +        str(theta_th[ii])) 

                # send to execution
                self.runCommands(commandList, "torso_head_controller")


    def __init__(self, _mode):  
        
        if _mode == ExtForce.Mode.Teaching:

            tau_th = [2.0, 2.0, 1.0, 1.0, 0.5, 0.5,            2.0, 2.0, 1.0, 1.0, 0.5, 0.5,      20.5, 20.5, 20.5, 20.5]
            kp = [0.1, 0.1, 0.05, 0.05, 0.1, 0.1,            0.1, 0.1, 0.05, 0.05, 0.1, 0.1,        0.0,0.0,0.0,0.0]
            sum_e_max = [200.0,200.0,100.0,100.0,50.0,50.0,  200.0,200.0,100.0,100.0,50.0,50.0,   200.0,200.0,50.0,50.0]
            d = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
            # kr = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
            kr = [0.1]*16
            kr[5] = 0.02
            kr[11] = 0.02
            theta_th = [1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57]
            theta_th = [i*1.63 for i in theta_th]

            self.registerParameters( tau_th, kp, sum_e_max, d, kr, theta_th)


        elif _mode == ExtForce.Mode.Experiment:

            tau_th = [12.0,12.0,5.0,5.0,1.5,1.5,12.0,12.0,5.0,5.0,1.5,1.5,200.0,200.0,200.0,200.0]
            kp = [0.025,0.025,0.05,0.05,0.1,0.1,0.025,0.025,0.05,0.05,0.1,0.1,0.025,0.025,0.05,0.05]
            sum_e_max = [200.0,200.0,100.0,100.0,50.0,50.0,200.0,200.0,100.0,100.0,50.0,50.0,200.0,200.0,50.0,50.0]
            d = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
            kr = [0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8,0.8]
            # kr = [0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
            theta_th = [1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57,1.57]
            theta_th = [i*1.63 for i in theta_th]

            self.registerParameters( tau_th, kp, sum_e_max, d, kr, theta_th)
            

            
def gostart(torobo):
    """ GO TO STARTING POSITION """
    idx = 4
    TIME = 5
    try:
        with HiddenPrints():
            servo_on(torobo)
            set_softness_override(torobo, 10.0)
            set_velocity_override(torobo, 10.0)
            with open('rad.pickle', 'rb') as f:
                RAD = pickle.load(f)

            left_arm_positions = RAD[idx][0:6]
            right_arm_positions = RAD[idx][6:12]
            torso_head_positions = RAD[idx][12:16]

            torobo.move(ToroboOperator.LEFT_ARM, positions=left_arm_positions, duration=TIME)
            torobo.move(ToroboOperator.RIGHT_ARM, positions=left_arm_positions, duration=TIME)
            torobo.move(ToroboOperator.TORSO_HEAD, positions=torso_head_positions, duration=TIME)
            rospy.sleep(TIME+0.1)
        print(colored('SUCCESS', 'white', 'on_green'))
        print (torso_head_positions)
    except:
        print('ERROR')
    

def setModeOverrides(torobo):
    """SET THE NEEDED KINEMATIC PARAMETERS AND MODES, OVERRIDES ON THE RIGHT JOINTS"""
    print('SETTING MODE: set_external_force_following_online_trajectory_control')
    with HiddenPrints():
        set_external_force_following_online_trajectory_control(torobo)
        torobo.set_control_mode(ToroboOperator.TORSO_HEAD, ['torso_head/joint_1',
                                                            'torso_head/joint_2'],
                                'position')
    print('SETTING DYNAMIC PARAMETERS')
    with HiddenPrints():
        time.sleep(1)
        ExtForce(ExtForce.Mode.Teaching)
        time.sleep(1)
    print('OVERRIDES SET')
    with HiddenPrints():
        set_softness_override(torobo, 10.0)
        set_velocity_override(torobo, 10.0)


exp = Experiment()
torobo = ToroboOperator()
gostart(torobo)
setModeOverrides(torobo)

# start video publisher

if not 'bridge' in locals():
    bridge = CvBridge()
if not 'torobo' in locals():
    torobo = ToroboOperator()
if not 'imagebuff' in locals():
    imagebuff = 0
    def image_callback(msg):
        global imagebuff
        imagebuff = bridge.imgmsg_to_cv2(msg, "rgb8") 
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    time.sleep(0.5)

# start the YOLO video tracker
tracker = Tracker()
tracker.start()

INTERVAL = 0.5

zero_vel = [0.0 for i in range(ALL_JOINTS)]
cur_pos, cur_vel = get_cur_joints(torobo)
tgt_pos = copy.deepcopy(cur_pos)


start_time = rospy.get_time()
keepGoing = True
posWinBuffer = deque(maxlen=exp.windowSize)    
step = 0

cur_pos_list = [] # current position
tgt_pos_list = [] # target position
times_list = [] # processing time
state_list = [] # network states
gravity_compensation_effort_list = []
ref_dynamics_effort_list = []
cur_dynamics_effort_list = []
inertia_diagonal_list = []
opt_elbo_list = []        

vTime = 0.0
cumSleepTime = 0        

tgt_pos_buffer = np.zeros((exp.nActDof,), dtype=float);
dataOut = (ctypes.c_float * exp.nActDof)(*tgt_pos_buffer)

elbo_buffer = np.zeros((3,), dtype=float);
elboOut = (ctypes.c_float * 3)(*elbo_buffer)

#m_state = np.zeros((exp.stateBufferSize,), dtype=float)
#m_stateOut = (ctypes.c_float * exp.stateBufferSize)(*m_state)

while not (rospy.is_shutdown()) and keepGoing:        

    start_time = rospy.get_time()
    echo("Seq step = {}".format(step+1))      

    cur_pos, cur_vel = get_cur_joints(torobo)               

    # generate the robot intention        
    tgt_pos_buffer = np.zeros((exp.nActDof,), dtype=float);
    dataOut = (ctypes.c_float * exp.nActDof)(*tgt_pos_buffer)            

    exp.nrl.e_generate(dataOut)                                        
    tgt_pos = np.frombuffer(dataOut, np.float32).tolist() 

    vTime += 1.0

    opt_elbo = [0.0,0.0,0.0]
    
    if (exp.postdiction_enable):
        joints = exp.filterMask(cur_pos)
        posWinBuffer.append(np.array(joints))

        if (step >= exp.ERStartTime): 

            pos_win_1d = np.hstack(posWinBuffer)                
            exp.nrl.e_postdict((ctypes.c_float * exp.nBuffdata)(*pos_win_1d), elboOut, exp.stdoutPostiction)
            vTime = 0.0                    
            opt_elbo = np.frombuffer(elboOut, np.float32).tolist()                    
    
   

    cur_pos, cur_vel = get_cur_joints(torobo) 
    
    ######################################################3
    tgt_arr = np.array(tgt_pos)    
    crr_arr = np.array(cur_pos[0:exp.nActDof])
    tgt_arr = exp.kr*(tgt_arr - crr_arr) + crr_arr
    tgt_pos = tgt_arr.tolist()
    ######################################################
    

    tgt_pos_16 = np.radians(tgt_pos[0:12] + [0,0,0,0])
    print(tgt_pos_16)
    
#     move_pos_whole_body(torobo, tgt_pos_16, zero_vel)
    torobo.move(ToroboOperator.LEFT_ARM, positions=tgt_pos_16[0:6], duration=INTERVAL)
    torobo.move(ToroboOperator.RIGHT_ARM, positions=tgt_pos_16[6:12], duration=INTERVAL)

    # saving current state
    m_state = np.zeros((exp.stateBufferSize,), dtype=float)
    m_stateOut = (ctypes.c_float * exp.stateBufferSize)(*m_state)
    exp.nrl.e_getState(m_stateOut)
    st_data = np.frombuffer(m_stateOut, np.float32)

    exp.state_list.append(st_data)
    opt_elbo_list.append(opt_elbo)                                     
    tgt_pos_list.append(tgt_pos)
    cur_pos_list.append(cur_pos)                        

    step += 1               
    if(step == exp.T):        
        keepGoing = False

    # sleep             
    timePassed = (rospy.get_time() - start_time)
    if timePassed < 0.0:
         cumSleepTime -= timePassed
    else:
         cumSleepTime = 0.0

    sleep_time = INTERVAL - timePassed + cumSleepTime
    #sleep_time = 0.01
#     print("time passed: ", timePassed, " sleepTime: ", sleep_time)
    if(sleep_time  > 0.0):
        rospy.sleep(sleep_time)
    times_list.append([timePassed, INTERVAL])

np.save(exp.expPath + "/cur_pos", cur_pos_list)        
np.save(exp.expPath + "/tgt_pos", tgt_pos_list)       # target position
#np.save(exp.expPath + "/states", state_list)
np.save(exp.expPath + "/times", times_list)
np.savetxt(exp.expPath + "/states.csv", exp.state_list, delimiter=",")
#np.save(exp.expPath + "/dyn_gra_com_eff", gravity_compensation_effort_list)
#np.save(exp.expPath + "/dyn_ref_eff", ref_dynamics_effort_list)
#np.save(exp.expPath + "/dyn_cur_eff", cur_dynamics_effort_list)
#np.save(exp.expPath + "/dyn_ine_dia", inertia_diagonal_list)        
np.save(exp.expPath + "/opt_elbo", opt_elbo_list)        


# Moving to start position

with HiddenPrints:
    rospy.sleep(1)        
    gostart(torobo)
    rospy.sleep(3)
    servo_off(torobo)
    rospy.sleep(2)


