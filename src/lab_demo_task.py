#!/usr/bin/env python2.7

import os
import sys
import time
import copy
sys.path.insert(0, '/home/torobo/catkin_ws/src/torobo_robot/torobo_rnn/scripts')
# from utils import Utils
from torobo_rnn_utils import *
import rospy
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from NRL import NRL
from inspect import getmembers, isfunction
import ctypes

from adjustExtForce_v3Controller import ExtForce

class Experiment(object):
    
    def __init__(self):
        
        self.nrl = NRL()
        self.cwd="/home/torobo/catkin_ws/src/tutorial/PVRNN/scripts"
        # self.ut = Utils(self.cwd) 
        self.success = True

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
        
        self.actJointMask = [True,True,True,True,True,True,True,True,True,True,True,True,False,False,False,False]        

        self.forceMode = ExtForce.Mode.Experiment

        self.modelId  = "W_0_001"
        
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

        dataset = 'training_12dof'
        subject = 'test_1'

        self.expPath = self.cwd +"/data/experiment/{}".format(subject)
        self.prop_path = self.cwd + "/data/config/properties_{}.d".format(self.modelId)
        print(self.prop_path)
                
        if self.postdiction_enable:
            try:                        
                if (not os.path.exists(self.expPath)):
                    os.mkdir(self.expPath)                                    
                if self.postdiction_enable and self.simulation:                
                    try:
                        #self.bottomUpData = np.load(self.cwd + bottomUpDataPath) # simulation Only  
                        #My version
                        self.bottomUpData=np.loadtxt(bottomUpDataPath, delimiter=",")
                    except IOError:
                        self.sucess = False        
                        print ("The file [" + self.cwd + bottomUpDataPath + "] failed lo load")
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
       
       
        
        try:
            #prop_path = "/home/shpurov/catkin_ws/src/reaching_task/scripts/data/config/properties_W_0_001.d"
            prop_path="/home/torobo/catkin_ws/src/tutorial/src/data/config/modxp.d"
            self.nrl.newModel(prop_path.encode('ascii'))   
            self.nrl.load()
            nZ = '4,1'
            nD = '40,10'             

            print("Experiment Parameters: ")
            print("######################################################")
            print("model:                     {}".format(self.modelId))
            print("dataset:                   {}".format(dataset))
            print("Controller forceMode:      {}".format(self.forceMode))
            print("topDownId:                 {}".format(self.topDownId))            
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

        except:
            print("An error occurred, experiment terminated!")
            self.sucess = False        
            return
                            
                
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

    def main(self):
            
        torobo = ToroboOperator()
        
        zero_vel = [0.0 for i in range(ALL_JOINTS)]
        set_velocity_override(torobo, 30.0)
        servo_on(torobo)
        rospy.sleep(1)
        move_startpos(torobo)
        set_external_force_following_online_trajectory_control(torobo)
                
        cur_pos, cur_vel = get_cur_joints(torobo)
        
        tgt_pos = copy.deepcopy(cur_pos)
        
        if (not self.simulation):
            ExtForce(self.forceMode)        
                
        start_time = rospy.get_time()
            
        keepGoing = True
        
        posWinBuffer = deque(maxlen=self.windowSize)    
        
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
        
        tgt_pos_buffer = np.zeros((self.nActDof,), dtype=float);
        dataOut = (ctypes.c_float * self.nActDof)(*tgt_pos_buffer)
        
        elbo_buffer = np.zeros((3,), dtype=float);
        elboOut = (ctypes.c_float * 3)(*elbo_buffer)
        
        #m_state = np.zeros((self.stateBufferSize,), dtype=float)
        #m_stateOut = (ctypes.c_float * self.stateBufferSize)(*m_state)
                
        while not (rospy.is_shutdown()) and keepGoing:        

            start_time = rospy.get_time()
            print ("Seq step = {}".format(step+1))      

            cur_pos, cur_vel = get_cur_joints(torobo)               

            # generate the robot intention        
            tgt_pos_buffer = np.zeros((self.nActDof,), dtype=float);
            dataOut = (ctypes.c_float * self.nActDof)(*tgt_pos_buffer)            

            self.nrl.e_generate(dataOut)                                        
            tgt_pos = np.frombuffer(dataOut, np.float32).tolist() 
            #print(tgt_pos)
                        
            vTime += 1.0

            opt_elbo = [0.0,0.0,0.0]                            
            if (self.postdiction_enable):
                if (self.simulation):                 
                    if self.simInteration:                       
                        #posWinBuffer.append(np.array(self.bottomUpData[step][0:self.nActDof])) # <------- Simulation
                        #joints = self.filterMask(self.bottomUpData[step])
                        joints = self.bottomUpData[step]
                        posWinBuffer.append(np.array(joints)) # <------- Simulation

                    else: 
                        #posWinBuffer.append(np.array(cur_pos[0:self.nActDof]))
                        joints = self.filterMask(cur_pos)
                        posWinBuffer.append(np.array(joints))
                else:
                    #posWinBuffer.append(np.array(cur_pos[0:self.nActDof])) # <------- Experiment                
                    joints = self.filterMask(cur_pos)
                    posWinBuffer.append(np.array(joints))
                                
                if (step >= self.ERStartTime): 
                    
                    pos_win_1d = np.hstack(posWinBuffer)                
                    self.nrl.e_postdict((ctypes.c_float * self.nBuffdata)(*pos_win_1d), elboOut, self.stdoutPostiction)
                    vTime = 0.0                    
                    opt_elbo = np.frombuffer(elboOut, np.float32).tolist()                    
                                        
            
            cur_pos, cur_vel = get_cur_joints(torobo) 
            #gravity_compensation_effort, ref_dynamics_effort, cur_dynamics_effort, inertia_diagonal = get_robot_dynamics(torobo)
    
            if self.simulation:
                if self.postdiction_enable and self.moveSimInteration:
                    tgt_pos  = posWinBuffer[-1].tolist();
                else:
                    tgt_arr = np.array(tgt_pos)    
                    crr_arr = np.array(cur_pos[0:self.nActDof])
                    tgt_arr = self.kr*(tgt_arr - crr_arr) + crr_arr
                    tgt_pos = tgt_arr.tolist()                        
                    
                                    
            
            tgt_pos_16 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]+ tgt_pos[0:12] + [0.0,0.0,0.0,-15.0]
            #print(tgt_pos_16)
            
            move_pos_whole_body(torobo, tgt_pos_16, zero_vel)
    
            # saving current state
            m_state = np.zeros((self.stateBufferSize,), dtype=float)
            m_stateOut = (ctypes.c_float * self.stateBufferSize)(*m_state)	
            self.nrl.e_getState(m_stateOut)
            st_data = np.frombuffer(m_stateOut, np.float32)
            
            self.state_list.append(st_data)
            opt_elbo_list.append(opt_elbo)                                     
            tgt_pos_list.append(tgt_pos)
            cur_pos_list.append(cur_pos)                        
            #gravity_compensation_effort_list.append(gravity_compensation_effort)
            #ref_dynamics_effort_list.append(ref_dynamics_effort)
            #cur_dynamics_effort_list.append(cur_dynamics_effort)
            #inertia_diagonal_list.append(inertia_diagonal)
    
            step += 1               
            if(step == self.T):        
                keepGoing = False
    
            # sleep             
            timePassed = (rospy.get_time() - start_time)
            if timePassed < 0.0:
                 cumSleepTime -= timePassed
            else:
                 cumSleepTime = 0.0
             
            sleep_time = INTERVAL - timePassed + cumSleepTime
            #sleep_time = 0.01
            print("time passed: ", timePassed, " sleepTime: ", sleep_time)
            if(sleep_time  > 0.0):
                rospy.sleep(sleep_time)
            times_list.append([timePassed, INTERVAL])
    
        np.save(self.expPath + "/cur_pos", cur_pos_list)        
        np.save(self.expPath + "/tgt_pos", tgt_pos_list)       # target position
        #np.save(self.expPath + "/states", state_list)
        np.save(self.expPath + "/times", times_list)
        np.savetxt(self.expPath + "/states.csv", self.state_list, delimiter=",")
        #np.save(self.expPath + "/dyn_gra_com_eff", gravity_compensation_effort_list)
        #np.save(self.expPath + "/dyn_ref_eff", ref_dynamics_effort_list)
        #np.save(self.expPath + "/dyn_cur_eff", cur_dynamics_effort_list)
        #np.save(self.expPath + "/dyn_ine_dia", inertia_diagonal_list)        
        np.save(self.expPath + "/opt_elbo", opt_elbo_list)        

       
        # Moving to start position    
        rospy.sleep(1)        
        move_startpos(torobo)
        rospy.sleep(3)
        servo_off(torobo)
        rospy.sleep(2)
    

exp = Experiment()
if exp.success:
    exp.main()        
