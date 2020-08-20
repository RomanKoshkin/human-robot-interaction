#!/usr/bin/env python

from torobo_rnn_utils import *
import ctypes
import rospy

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

def next_step_limit_cycle(cur_pos):
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
    
    for i in range(ALL_JOINTS):
        tgt_pos[i] = (rescale(tgt_pos[i], -1.0, 1.0, scale_min[i], scale_max[i]))*multiplier[i] + offset[i]
        # tgt_vel[i] = (cur_pos[i]-tgt_pos[i])/INTERVAL
        tgt_vel[i] = 0.0

    print "Scaled output from RNN"
    print (tgt_pos)
    
    return tgt_pos, tgt_vel

    // Parameter settings
    float tau = 10.0f;
    float a = 0.8f;
    float b = 0.3f;
    float scaleFactor = 30.0f;
    float j2Offset = 20.0f;
    float j4Offset = 0.0f;

    // Get current position
    Common.Master.Com.CRecvData armRecv = arm.GetRecvData();
    float j2_t = armRecv.joint[1].position;
    float j4_t = armRecv.joint[3].position;

    // Offset & Scaling
    float x_t = (j2_t - j2Offset) / scaleFactor;
    float y_t = (j4_t - j4Offset) / scaleFactor;

    // Dynamics function
    float x_tp1 = 1.0f / tau * y_t + x_t;
    float y_tp1 = 1.0f / tau * (-a * (x_t * x_t - b) * y_t - x_t) + y_t;

    // Offset & Scaling
    float j2_tp1 = x_tp1 * scaleFactor + j2Offset;
    float j4_tp1 = y_tp1 * scaleFactor + j4Offset;

    // Set ref position to arm joints
    Common.Master.Com.CSendData sendData = new Common.Master.Com.CSendData(arm._jointsNum);
    string[] jointNameArray = new string[] { "2", "4" };
    arm.SetJointOrder(jointNameArray, Common.Master.Enum.ePacketOrder.POSITION, ref sendData);
    arm.SetValue(jointNameArray[0], 1, j2_tp1, ref sendData);
    arm.SetValue(jointNameArray[1], 1, j4_tp1, ref sendData);

    float v2 =( j2_tp1 - j2_t) / 0.3f;
    float v4 = (j4_tp1 - j4_t) / 0.3f;
    arm.SetValue(jointNameArray[0], 2, v2, ref sendData);
    arm.SetValue(jointNameArray[1], 2, v4, ref sendData);

    arm.SendDataToMCU(ref sendData);

def main():
    torobo = ToroboOperator()
    #move_homepos(torobo)
    step = 0
    cur_pos, cur_vel = get_cur_joints(torobo)
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        cur_pos, cur_vel = get_cur_joints(torobo)
        tgt_pos, tgt_vel = next_step_limit_cycle(cur_pos)
        move_pos(torobo, tgt_pos, tgt_vel)
        step += 1
        rate.sleep()
        
if __name__ == "__main__":
    main()

