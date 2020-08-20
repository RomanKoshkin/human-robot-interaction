#!/usr/bin/env python

import os, sys
import math
import rospy
from torobo_operator__upd    import ToroboOperator

from multiprocessing import Process
import threading
import time
import numpy as np

import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Robot parameters 
#INTERVAL = 0.02 # sec
#INTERVAL = 0.05 # sec
#INTERVAL = 0.05*7 # sec
#INTERVAL = 0.05*3 # sec
#INTERVAL = 0.05*4 # sec
#INTERVAL = 0.07 # sec
#INTERVAL = 0.1 # sec
INTERVAL = 0.10 # sec
CYCLIC_INTERVAL = rospy.Duration(secs=0, nsecs=INTERVAL*1000000000)
LIMIT_POS_DIFF = 20 # deg
LIMIT_VELOCITY = 20 # deg

ALL_JOINTS = 16
LARM_JOINTS = 6
RARM_JOINTS = 6
TOHD_JOINTS = 4 # waist roll, waist pitch, neck roll, neck pitch

# NB: All in degrees, ordered as: [ LEFT-ARM-JOINTS RIGHT-ARM-JOINTS TORSO-HEAD-JOINTS ]
# Scale output to within physical joint limits
# scale_min = [-70.0, -45.0, -160.0, -50.0, -160.0, -105.0, -70.0, -45.0, -160.0, -50.0, -160.0, -105.0, -160.0, -90.0, -90.0, -60.0]
# scale_max = [250.0, 105.0,  160.0, 115.0,  160.0,  105.0, 250.0, 105.0,  160.0, 115.0,  160.0,  105.0,  160.0,  90.0,   90.0, 45.0]
scale_min = [-65.0, -40.0, -155.0, -45.0, -155.0, -100.0, -65.0, -40.0, -155.0, -45.0, -155.0, -100.0, -155.0, -50.0, -85.0, -40.0]
scale_max = [245.0, 100.0,  155.0, 110.0,  155.0,  100.0, 245.0, 100.0,  155.0, 110.0,  155.0,  100.0,  155.0,  50.0,  85.0,  40.0]
# Offset and multiplier bypasses output scaling
offset =     [-20.0, 10.0, 0.0, 0.0, 0.0, 0.0, -20.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
multiplier = [  1.0,  1.0, 1.0, 1.0, 1.0, 1.0,   1.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

def rescale(inp, inpMin, inpMax, outMin, outMax):
    return (((outMax-outMin)*(inp-inpMin))/(inpMax-inpMin))+outMin

def array_radians(angles):
    for i in range(len(angles)):
        angles[i] = math.radians(angles[i])
    return angles

def array_degrees(angles):
    for i in range(len(angles)):
        angles[i] = math.degrees(angles[i])
    return angles

def servo_on(torobo):
    torobo.servo_on(ToroboOperator.LEFT_ARM, 'all')
    torobo.servo_on(ToroboOperator.RIGHT_ARM, 'all')
    torobo.servo_on(ToroboOperator.TORSO_HEAD, 'all')
    torobo.servo_on(ToroboOperator.LEFT_GRIPPER, 'all')
    torobo.servo_on(ToroboOperator.RIGHT_GRIPPER, 'all')

def servo_off(torobo):
    torobo.servo_off(ToroboOperator.LEFT_ARM, 'all')
    torobo.servo_off(ToroboOperator.RIGHT_ARM, 'all')
    torobo.servo_off(ToroboOperator.TORSO_HEAD, 'all')
    torobo.servo_off(ToroboOperator.LEFT_GRIPPER, 'all')
    torobo.servo_off(ToroboOperator.RIGHT_GRIPPER, 'all')

def set_position_control(torobo):
    torobo.set_control_mode(ToroboOperator.LEFT_ARM, 'all', 'position')
    torobo.set_control_mode(ToroboOperator.RIGHT_ARM, 'all', 'position')
    torobo.set_control_mode(ToroboOperator.TORSO_HEAD, 'all', 'position')
    torobo.set_control_mode(ToroboOperator.LEFT_GRIPPER, 'all', 'position')
    torobo.set_control_mode(ToroboOperator.RIGHT_GRIPPER, 'all', 'position')

def set_external_force_following_control(torobo):
    torobo.set_control_mode(ToroboOperator.LEFT_ARM, 'all', 'external_force_following')
    torobo.set_control_mode(ToroboOperator.RIGHT_ARM, 'all', 'external_force_following')
    torobo.set_control_mode(ToroboOperator.TORSO_HEAD, 'all', 'external_force_following')
    torobo.set_control_mode(ToroboOperator.LEFT_GRIPPER, 'all', 'external_force_following')
    torobo.set_control_mode(ToroboOperator.RIGHT_GRIPPER, 'all', 'external_force_following')

def set_online_trajectory_control(torobo):
    torobo.set_control_mode(ToroboOperator.LEFT_ARM, 'all', 'online_trajectory')
    torobo.set_control_mode(ToroboOperator.RIGHT_ARM, 'all', 'online_trajectory')
    torobo.set_control_mode(ToroboOperator.TORSO_HEAD, 'all', 'online_trajectory')
    torobo.set_control_mode(ToroboOperator.LEFT_GRIPPER, 'all', 'online_trajectory')
    torobo.set_control_mode(ToroboOperator.RIGHT_GRIPPER, 'all', 'online_trajectory')

def set_external_force_following_online_trajectory_control(torobo):
    torobo.set_control_mode(ToroboOperator.LEFT_ARM, 'all', 'external_force_following_online_trajectory')
    torobo.set_control_mode(ToroboOperator.RIGHT_ARM, 'all', 'external_force_following_online_trajectory')
    torobo.set_control_mode(ToroboOperator.TORSO_HEAD, 'all', 'external_force_following_online_trajectory')
    torobo.set_control_mode(ToroboOperator.LEFT_GRIPPER, 'all', 'external_force_following_online_trajectory')
    torobo.set_control_mode(ToroboOperator.RIGHT_GRIPPER, 'all', 'external_force_following_online_trajectory')

def set_velocity_override(torobo, override_value):
    # override_value: 0(slow) - 100(fast) [%]
    torobo.set_robot_controller_parameter(ToroboOperator.LEFT_ARM, 'velocity_override', [override_value], ['all'])
    torobo.set_robot_controller_parameter(ToroboOperator.RIGHT_ARM, 'velocity_override', [override_value], ['all'])
    torobo.set_robot_controller_parameter(ToroboOperator.TORSO_HEAD, 'velocity_override', [override_value], ['all'])

def set_softness_override(torobo, override_value):
    # override_value: 0(soft) - 100(hard) [%]
    torobo.set_robot_controller_parameter(ToroboOperator.LEFT_ARM, 'softness_override', [override_value], ['all'])
    torobo.set_robot_controller_parameter(ToroboOperator.RIGHT_ARM, 'softness_override', [override_value], ['all'])
    torobo.set_robot_controller_parameter(ToroboOperator.TORSO_HEAD, 'softness_override', [override_value], ['all'])
    
def move_startpos(torobo):
    print ("Moving to start position")
    set_position_control(torobo)
    torobo.move(ToroboOperator.LEFT_ARM, positions=[math.radians(90.0), math.radians(65.0), math.radians(-6.0), math.radians(45.0), math.radians(-90.0), math.radians(45.0)], duration=5.0)
    torobo.move(ToroboOperator.RIGHT_ARM, positions=[math.radians(100.0), math.radians(60.0), math.radians(10.0), math.radians(55.0), math.radians(-105.0), math.radians(-30.0)], duration=5.0)
    torobo.move(ToroboOperator.TORSO_HEAD, positions=[math.radians(0.0), math.radians(0.0),math.radians(0.0), math.radians(-15.0)], duration=5.0)
    rospy.sleep(5)

def move_homepos(torobo):
    print ("Moving to home position")
    set_position_control(torobo)
    torobo.move(ToroboOperator.LEFT_ARM, positions=[0.0, math.radians(90.0), 0.0, 0.0, 0.0, 0.0], duration=5.0)
    torobo.move(ToroboOperator.RIGHT_ARM, positions=[0.0, math.radians(90.0), 0.0, 0.0, 0.0, 0.0], duration=5.0)
    torobo.move(ToroboOperator.TORSO_HEAD, positions=[0.0, 0.0, 0.0, 0.0], duration=5.0)
    rospy.sleep(5)
    
def move_pos_time(torobo, pos, time):  
    print ("Moving to position " + str(pos))
    set_position_control(torobo)
    torobo.move(ToroboOperator.LEFT_ARM, positions=[math.radians(pos[0]), math.radians(pos[1]), math.radians(pos[2]), math.radians(pos[3]), math.radians(pos[4]), math.radians(pos[5])], duration=time)
    torobo.move(ToroboOperator.RIGHT_ARM, positions=[math.radians(pos[6]), math.radians(pos[7]), math.radians(pos[8]), math.radians(pos[9]), math.radians(pos[10]), math.radians(pos[11])], duration=time)
    torobo.move(ToroboOperator.TORSO_HEAD, positions=[math.radians(pos[12]), math.radians(pos[13]), math.radians(pos[14]), math.radians(pos[15])], duration=time)
    rospy.sleep(time)

def move_pos(torobo, tgt_pos, tgt_vel):
    torobo.move(
        ToroboOperator.LEFT_ARM,
        positions=[math.radians(tgt_pos[0]), math.radians(tgt_pos[1]), math.radians(tgt_pos[2]), math.radians(tgt_pos[3]), math.radians(tgt_pos[4]), math.radians(tgt_pos[5])],
        velocities=[math.radians(tgt_vel[0]), math.radians(tgt_vel[1]), math.radians(tgt_vel[2]), math.radians(tgt_vel[3]), math.radians(tgt_vel[4]), math.radians(tgt_vel[5])],
        duration=INTERVAL
    )
    torobo.move(
        ToroboOperator.RIGHT_ARM,
        positions=[math.radians(tgt_pos[6]), math.radians(tgt_pos[7]), math.radians(tgt_pos[8]), math.radians(tgt_pos[9]), math.radians(tgt_pos[10]), math.radians(tgt_pos[11])],
        velocities=[math.radians(tgt_vel[6]), math.radians(tgt_vel[7]), math.radians(tgt_vel[8]), math.radians(tgt_vel[9]), math.radians(tgt_vel[10]), math.radians(tgt_vel[11])],
        duration=INTERVAL
    )
    torobo.move(
        ToroboOperator.TORSO_HEAD,
        positions=[math.radians(tgt_pos[12]), math.radians(tgt_pos[13]), math.radians(tgt_pos[14]), math.radians(tgt_pos[15])],
        velocities=[math.radians(tgt_vel[12]), math.radians(tgt_vel[13]), math.radians(tgt_vel[14]), math.radians(tgt_vel[15])],
        duration=INTERVAL
    )
    # rospy.sleep(INTERVAL)


def move_pos_whole_body(torobo, tgt_pos, tgt_vel):
    torobo.move_joints( left_arm_joint_names=torobo.get_joint_names(ToroboOperator.LEFT_ARM),
                        left_arm_positions=[math.radians(tgt_pos[0]), math.radians(tgt_pos[1]), math.radians(tgt_pos[2]), math.radians(tgt_pos[3]), math.radians(tgt_pos[4]), math.radians(tgt_pos[5])],
                        left_arm_velocities=[math.radians(tgt_vel[0]), math.radians(tgt_vel[1]), math.radians(tgt_vel[2]), math.radians(tgt_vel[3]), math.radians(tgt_vel[4]), math.radians(tgt_vel[5])],
                        right_arm_joint_names=torobo.get_joint_names(ToroboOperator.RIGHT_ARM),
                        right_arm_positions=[math.radians(tgt_pos[6]), math.radians(tgt_pos[7]), math.radians(tgt_pos[8]), math.radians(tgt_pos[9]), math.radians(tgt_pos[10]), math.radians(tgt_pos[11])],
                        right_arm_velocities=[math.radians(tgt_vel[6]), math.radians(tgt_vel[7]), math.radians(tgt_vel[8]), math.radians(tgt_vel[9]), math.radians(tgt_vel[10]), math.radians(tgt_vel[11])],
                        torso_head_joint_names=torobo.get_joint_names(ToroboOperator.TORSO_HEAD),
                        torso_head_positions=[math.radians(tgt_pos[12]), math.radians(tgt_pos[13]), math.radians(tgt_pos[14]), math.radians(tgt_pos[15])],
                        torso_head_velocities=[math.radians(tgt_vel[12]), math.radians(tgt_vel[13]), math.radians(tgt_vel[14]), math.radians(tgt_vel[15])],
                        duration=INTERVAL)

def get_cur_joints(torobo):
    la_pos, la_vel, _ = torobo.get_joint_states(ToroboOperator.LEFT_ARM)
    ra_pos, ra_vel, _ = torobo.get_joint_states(ToroboOperator.RIGHT_ARM)
    th_pos, th_vel, _ = torobo.get_joint_states(ToroboOperator.TORSO_HEAD)
    cur_pos = [0.0 for i in range(ALL_JOINTS)]
    cur_vel = [0.0 for i in range(ALL_JOINTS)]
    for i in range(ALL_JOINTS):
        if i >= 0 and i < LARM_JOINTS:
            cur_pos[i] = la_pos[i]
            cur_vel[i] = la_vel[i]
        elif i >= LARM_JOINTS and i < LARM_JOINTS + RARM_JOINTS:
            cur_pos[i] = ra_pos[i-LARM_JOINTS]
            cur_vel[i] = ra_vel[i-LARM_JOINTS]
        elif i >= LARM_JOINTS + RARM_JOINTS and i < LARM_JOINTS + RARM_JOINTS + TOHD_JOINTS:
            cur_pos[i] = th_pos[i-(LARM_JOINTS + RARM_JOINTS)]
            cur_vel[i] = th_vel[i-(LARM_JOINTS + RARM_JOINTS)]
    
    return array_degrees(cur_pos), array_degrees(cur_vel)

def get_cur_joints_torque(torobo):
    _, _, la_eft = torobo.get_joint_states(ToroboOperator.LEFT_ARM)
    _, _, ra_eft = torobo.get_joint_states(ToroboOperator.RIGHT_ARM)
    _, _, th_eft = torobo.get_joint_states(ToroboOperator.TORSO_HEAD)
    cur_eft = [0.0 for i in range(ALL_JOINTS)]
    for i in range(ALL_JOINTS):
        if i >= 0 and i < LARM_JOINTS:
            cur_eft[i] = la_eft[i]
        elif i >= LARM_JOINTS and i < LARM_JOINTS + RARM_JOINTS:
            cur_eft[i] = ra_eft[i-LARM_JOINTS]
        elif i >= LARM_JOINTS + RARM_JOINTS and i < LARM_JOINTS + RARM_JOINTS + TOHD_JOINTS:
            cur_eft[i] = th_eft[i-(LARM_JOINTS + RARM_JOINTS)]
    
    return cur_eft

def get_robot_dynamics(torobo):
    la_gravity_compensation_effort, la_ref_dynamics_effort, la_cur_dynamics_effort, la_inertia_diagonal = torobo.get_robot_dynamics(ToroboOperator.LEFT_ARM)
    ra_gravity_compensation_effort, ra_ref_dynamics_effort, ra_cur_dynamics_effort, ra_inertia_diagonal = torobo.get_robot_dynamics(ToroboOperator.RIGHT_ARM) 
    th_gravity_compensation_effort, th_ref_dynamics_effort, th_cur_dynamics_effort, th_inertia_diagonal = torobo.get_robot_dynamics(ToroboOperator.TORSO_HEAD)  

    gravity_compensation_effort = [0.0 for i in range(ALL_JOINTS)]
    ref_dynamics_effort = [0.0 for i in range(ALL_JOINTS)]
    cur_dynamics_effort = [0.0 for i in range(ALL_JOINTS)]
    inertia_diagonal = [0.0 for i in range(ALL_JOINTS)]

    for i in range(ALL_JOINTS):
        if i >= 0 and i < LARM_JOINTS:
            gravity_compensation_effort[i] = la_gravity_compensation_effort[i] 
            ref_dynamics_effort[i] = la_ref_dynamics_effort[i] 
            cur_dynamics_effort[i] = la_cur_dynamics_effort[i]
            inertia_diagonal[i] = la_inertia_diagonal[i]
        elif i >= LARM_JOINTS and i < LARM_JOINTS + RARM_JOINTS:
            gravity_compensation_effort[i] = ra_gravity_compensation_effort[i-LARM_JOINTS] 
            ref_dynamics_effort[i] = ra_ref_dynamics_effort[i-LARM_JOINTS] 
            cur_dynamics_effort[i] = ra_cur_dynamics_effort[i-LARM_JOINTS]
            inertia_diagonal[i] = ra_inertia_diagonal[i-LARM_JOINTS]
        elif i >= LARM_JOINTS + RARM_JOINTS and i < LARM_JOINTS + RARM_JOINTS + TOHD_JOINTS:
            gravity_compensation_effort[i] = th_gravity_compensation_effort[i-(LARM_JOINTS + RARM_JOINTS)] 
            ref_dynamics_effort[i] = th_ref_dynamics_effort[i-(LARM_JOINTS + RARM_JOINTS)] 
            cur_dynamics_effort[i] = th_cur_dynamics_effort[i-(LARM_JOINTS + RARM_JOINTS)]
            inertia_diagonal[i] = th_inertia_diagonal[i-(LARM_JOINTS + RARM_JOINTS)]    
    return gravity_compensation_effort, ref_dynamics_effort, cur_dynamics_effort, inertia_diagonal

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def set_dynamics_params(dyn_params):
    def runCommands(_commandList, _controller):
        with HiddenPrints():
            for command in _commandList:
                torobo_easy_command.SendEasyCommandText(_controller, command)
                rospy.sleep(0.01)
    try:
        with HiddenPrints():
            tauth = dyn_params['tauth']
            kr = dyn_params['kr']
            kp = dyn_params['kp']
            sigmaemax = dyn_params['sigmaemax']
            damping = dyn_params['damping']
            thetath = dyn_params['thetath']

            dampingTorsoHead = dyn_params['dampingTorsoHead']
            tauthTorsoHead = dyn_params['tauthTorsoHead']
            sigmaemaxTorsoHead = dyn_params['sigmaemaxTorsoHead']
            thetathTorsoHead = dyn_params['thetathTorsoHead']
            krTorsoHead = dyn_params['krTorsoHead']
            kpTorsoHead = dyn_params['kpTorsoHead']
            
            armDof = 6
            torsoHeadDof = 4
                            
            for i in range(armDof):

                # constructing the commands for the left arm
                commandList = []
                commandList.append("param " + str(i+1) + " ffdamping " +     str(damping[i]))
                commandList.append("param " + str(i+1) + " fftauth " +       str(tauth[i]))
                commandList.append("param " + str(i+1) + " ffsigmaemax " +   str(sigmaemax[i]))
                commandList.append("param " + str(i+1) + " softki " +        str(thetath[i]))
                commandList.append("param " + str(i+1) + " ffkr " +          str(kr[i]))
                commandList.append("param " + str(i+1) + " ffkp " +          str(kp[i]))

                # send to execution
                runCommands(commandList, "left_arm_controller")

                # constructing the commands for the right arm
                commandList = []
                commandList.append("param " + str(i+1) + " ffdamping " +     str(damping[i]))
                commandList.append("param " + str(i+1) + " fftauth " +       str(tauth[i]))
                commandList.append("param " + str(i+1) + " ffsigmaemax " +   str(sigmaemax[i]))
                commandList.append("param " + str(i+1) + " softki " +        str(thetath[i]))
                commandList.append("param " + str(i+1) + " ffkr " +          str(kr[i]))
                commandList.append("param " + str(i+1) + " ffkp " +          str(kp[i]))

                # send to execution
                runCommands(commandList, "right_arm_controller")

            for i in range(torsoHeadDof):
                # constructing the commands for the head-torso chain
                commandList = []
                commandList.append("param " + str(i+1) + " ffdamping " +     str(dampingTorsoHead[i]))
                commandList.append("param " + str(i+1) + " fftauth " +       str(tauthTorsoHead[i]))
                commandList.append("param " + str(i+1) + " ffsigmaemax " +   str(sigmaemaxTorsoHead[i]))
                commandList.append("param " + str(i+1) + " softki " +        str(thetathTorsoHead[i]))
                commandList.append("param " + str(i+1) + " ffkr " +          str(krTorsoHead[i]))
                commandList.append("param " + str(i+1) + " ffkp " +          str(kpTorsoHead[i]))

                # send to execution
                runCommands(commandList, "torso_head_controller")
        
        print('SUCCESSFULLY RESET DYNAMIC PARAMETERS')
    except:
        print('RESETTING DYNAMIC PARAMETERS ___ FAILED ___')


class Recorder:
    '''
    RECORDING THE JOINTS, TORQUE AND dynamics AT THE SAME TIME. RECORDING RUNS IN A SEPARATE THREAD
    '''
    def __init__(self, torobo, suffix, SRATE):
        self.recording = False
        self.r = rospy.Rate(SRATE)
        self.torobo = torobo
        self.suffix = suffix
    
    def start(self):
        if self.recording == True:
            self.recording = False
            print('Killing the previous recorder')
            time.sleep(1.5)
        self.recording = True
        self.thread = threading.Thread(target=self.record_torque, args=([self.torobo, self.r], ))
        self.thread.daemon = True                           
        self.thread.start()
        print('Recording is {}'.format('ON' if self.thread.is_alive() else 'OFF'))
    
    def stop(self):
        self.recording = False
        time.sleep(1)
        print('Recording is {}'.format('ON' if self.thread.is_alive() else 'OFF'))
        
    def record_torque(self, args):
        torobo, r = args
        t = 0
        step = r.sleep_dur.to_sec()
        fname = "./joints/motor_rad_" + "angtor" + self.suffix + ".txt"
        if "motor_rad_"+ "angtor"+".txt" in os.listdir("./joints"):
            open(fname, 'w').close()
        if "EFF.dat" in os.listdir('./'):
            open("EFF" + self.suffix + ".dat", 'w').close()

        while self.recording:
            cur_pos_deg, _ = get_cur_joints(torobo)
            cur_pos_rad = array_radians(cur_pos_deg)
            linedat = cur_pos_rad + get_cur_joints_torque(torobo)
            line = ["{:.4f}\t".format(i) for i in linedat]
            line.append(str(t) + "\n")
            with open(fname, 'a') as f:
                f.writelines(line)

            linedat1 = []
            for sublist in list(torobo.get_robot_dynamics(ToroboOperator.LEFT_ARM)):
                for item in sublist:
                    linedat1.append(item)
            line1 = ["{:.4f}\t".format(i) for i in linedat1]
            line1.append(str(t) + "\n")
            with open("EFF" + self.suffix + ".dat", 'a') as f_eff:
                f_eff.writelines(line1)
            r.sleep()
            t += step


class CaptureVid:
    '''
    RECORDING VIDEO
    '''
    def __init__(self, SRATE):
        self.SRATE = SRATE
        self.image_topic = "/camera/color/image_raw"
        self.recording = False
    
    def start(self):
        if self.recording == True:
            self.recording = False
            print('Killing the previous recorder')
            time.sleep(1.5)
        self.recording = True
        self.thread = threading.Thread(target=self.record_video, args=([self.SRATE], ))
        self.thread.daemon = True                           
        self.thread.start()
        print('VIDEO Recording is {}'.format('ON' if self.thread.is_alive() else 'OFF'))
    
    def stop(self):
        self.recording = False
        time.sleep(1)
        print('VIDEO Recording is {}'.format('ON' if self.thread.is_alive() else 'OFF'))
    
    def record_video(self, args):
        self.bridge = CvBridge()
        self.fig = plt.figure()
        self.ax = self.fig.gca()
        self.fig.show()
        while self.recording:
            msg = rospy.wait_for_message(self.image_topic, Image)
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.ax.imshow(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
            self.fig.canvas.draw()

