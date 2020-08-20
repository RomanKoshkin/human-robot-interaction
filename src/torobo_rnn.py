#!/usr/bin/env python

from torobo_rnn_utils import *
from rnn_state_publisher import *
import ctypes
import rospy
import copy

# RNN parameters
rnn_path = os.path.dirname(__file__) + "/mtrnn/"
lib = ctypes.cdll.LoadLibrary(rnn_path + "/lib/libmtrnn.so")
rnn = lib.rnet_new(30, rnn_path + "/lib/")
lib.rnet_settrainfp(rnn, rnn_path + "/input/16shybrid_train")
lib.rnet_settargetfp(rnn, rnn_path + "input/16shybrid_target")
lib.rnet_countinput(rnn)
lib.rnet_counttarget(rnn)
lib.rnet_loadepoch(rnn)
rnn_steps = 400

def next_step_rnn(cur_pos):
    """Get the next joint target position from current position using an RNN (library call)
         @cur_pos: array of joint angles in deg (format: [ LEFT-ARM RIGHT-ARM TORSO ])
    """
    cur_pos_scaled = [rescale((cur_pos[i]-offset[i])/multiplier[i], scale_min[i], scale_max[i], -1.0, 1.0) for i in range(len(cur_pos))]
    tgt_pos = cur_pos_scaled[:]
    tgt_vel = [0.0 for i in range(ALL_JOINTS)]
    
    print "Unscaled input to RNN"
    print (cur_pos)
    
    # Convert from python to C array
    rnn_cur_pos = (ctypes.c_double * len(cur_pos_scaled))(*cur_pos_scaled)
    rnn_tgt_pos = (ctypes.c_double * len(tgt_pos))(*tgt_pos)
        
    lib.rnet_openonline(rnn, rnn_cur_pos, rnn_tgt_pos, ALL_JOINTS, rnn_steps)
    
    # Convert back to python array
    rnn_tgt_pos_ptr = ctypes.cast(rnn_tgt_pos, ctypes.POINTER(ctypes.c_double))
    tgt_pos = [rnn_tgt_pos_ptr[i] for i in range(len(tgt_pos))]

    # Copy RNN input and output
    rnn_input = copy.deepcopy(cur_pos_scaled)
    rnn_output = copy.deepcopy(tgt_pos)

    for i in range(ALL_JOINTS):
        tgt_pos[i] = (rescale(tgt_pos[i], -1.0, 1.0, scale_min[i], scale_max[i]))*multiplier[i] + offset[i]
        tgt_vel[i] = (tgt_pos[i]-cur_pos[i])/INTERVAL
        # tgt_vel[i] = (tgt_pos[i]-cur_pos[i])/0.7
        # tgt_vel[i] = 0.0

    print "Scaled output from RNN"
    print (tgt_pos)
    
    return tgt_pos, tgt_vel, rnn_input, rnn_output

def main():
    torobo = ToroboOperator()

    # Set velocity override here (0-100[%])
    set_velocity_override(torobo, 30.0)
    # Set softness override here (0-100[%])
    set_softness_override(torobo, 100.0)

    # Set general output register
    set_general_output_register(torobo, 0, "ffthetaref")      # general_0 is ff_theta_ref
    set_general_output_register(torobo, 1, "ffthetastar")     # general_1 is ff_theta_star
    set_general_output_register(torobo, 2, "ffsigmae")        # general_2 is ff_sigma_e

    # !!!! Servo ON !!!!
    servo_on(torobo)
    rospy.sleep(1)

    # Move to start position
    move_startpos(torobo)

    # Set external force following online trajectory control mode
    set_external_force_following_online_trajectory_control(torobo)
    # Create publisher
    rnn_state_publisher = RnnStatePublisher(torobo)

    step = 0
    cur_pos, cur_vel = get_cur_joints(torobo)
    start_pos = copy.deepcopy(cur_pos)
    last_pos = copy.deepcopy(start_pos)
    
    cur_vel_rad = [0.0] * ALL_JOINTS
    zero_vel = [0.0] * ALL_JOINTS
    start_time = rospy.get_time()

    # In order to skip joints, for each joint copy last pos to cur pos, get next step, then copy start pos to target pos
    # Joint index 0-5 (L-Arm J1 to J6) 6-11 (R-Arm J1 to J6) 12-15 (Torso-Head)
    skip_joints =  [12]
    scale_joints = [4,5,10,11]
    scale_factor = 0.2
    for i in scale_joints:
        cur_pos[i] = cur_pos[i]*scale_factor

    while not rospy.is_shutdown():
        cur_pos, cur_vel = get_cur_joints(torobo)

        for i in skip_joints:
            cur_pos[i] = last_pos[i]
        for i in scale_joints:
            cur_pos[i] = cur_pos[i]/scale_factor
        
        tgt_pos, tgt_vel, rnn_input, rnn_output = next_step_rnn(cur_pos)
 
        last_pos = copy.deepcopy(tgt_pos)
        for i in skip_joints:
            tgt_pos[i] = start_pos[i]
        
        for i in scale_joints:
            tgt_pos[i] = tgt_pos[i]*scale_factor

        move_pos_whole_body(torobo, tgt_pos, zero_vel)

        # sim
        # for i in range(ALL_JOINTS):
        #     cur_vel_rad[i] = math.radians((cur_pos[i] - last_pos[i]) / INTERVAL)
        # real
        cur_vel_rad = map(math.radians, cur_vel)
        tgt_vel_rad = map(math.radians, tgt_vel)

        # publish rnn state
        rnn_state_publisher.publish(rnn_input, rnn_output, cur_vel_rad, tgt_vel_rad)

        # sleep
        cur_time = rospy.get_time()
        diff_time = cur_time - start_time
        sleep_time = INTERVAL - diff_time
        if(sleep_time > 0.0):
            rospy.sleep(sleep_time)
        start_time = rospy.get_time()

        step += 1

    # Servo off
    rospy.sleep(2)
    servo_off(torobo)
    rospy.sleep(1)

if __name__ == "__main__":
    main()
