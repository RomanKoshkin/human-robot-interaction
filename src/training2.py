import os
import sys

import ctypes
from NRL import NRL
import numpy as np



class Training(object):
    def __init__(self):
       
        print('Program begin')        
        
        # path to config file
#         prop_path = "/home/torobo/catkin_ws/src/tutorial/src/data/config/" + "modxp.d"
        # prop_path = "/home/torobo/catkin_ws/src/tutorial/src/data/config/" + "modxp5.d"
        prop_path = "/home/torobo/catkin_ws/src/tutorial/src/data/config/" + "modxp_long.d" 

        # path to where the model will be stored
#         model_path = "/home/torobo/catkin_ws/src/tutorial/src/data/model/" + "modxp"
        # model_path = "/home/torobo/catkin_ws/src/tutorial/src/data/model/" + "modxp5"
        model_path = "/home/torobo/catkin_ws/src/tutorial/src/data/model/" + "modxp_long"

        
      
       
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
        nTimes = 550

        # train for nTimes * modulus times
               
        for _ in range(nTimes):                
                nrl.t_loop(trainOut, modulus)
                e_sum = e_sum + modulus
           
        nrl.t_end()
        print('program finished')

train = Training()