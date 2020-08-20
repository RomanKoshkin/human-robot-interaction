#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:32:58 2020

@author: hseldon
"""

import os
import sys
# os.chdir('/home/shpurov/catkin_ws/src/reaching_task/scripts')
import ctypes
from NRL import NRL
import numpy as np
# print ("Please define model property file")
# property_file = str(input())
# print("Please define file to save the model")
# model_location = str(input())
class Training(object):
    def __init__(self):
       
        print('Program begin')        
        prop_path = "/home/torobo/catkin_ws/src/tutorial/src/data/config/" + "model_0_1_0_1_0_0_1_1_w_0.1.d"
        model_path = "/home/torobo/catkin_ws/src/tutorial/src/data/model/" + "model_0_1_0_1_0_0_1_1_w_0._1"

        
       
        print(model_path)
        try:                        
            if not os.path.exists(prop_path):
                print ("Error: the configuration file [%s] does not exist" % prop_path)
                return
            if not os.path.exists(model_path):
                os.mkdir(model_path)                                    
        except OSError:                
            print ("Error: creation of the directory [%s] failed" % model_path)
            return
                           
        nrl = NRL()
        print ("ok")
        nrl.newModel(prop_path.encode('ascii'))  
        print('model created')
        nrl.t_init(True)
        print('training initialized')
        train_buffer = np.zeros((7,), dtype=float)
        trainOut = (ctypes.c_float * 7)(*train_buffer)
        e_sum = 0
        modulus = 100
        nTimes = 300

        # train for nTimes * modulus times
               
        for _ in range(nTimes):                
                nrl.t_loop(trainOut, modulus)
                e_sum = e_sum + modulus
           
        nrl.t_end()
        print('program finished')

train = Training()        
