#!/usr/bin/env python

from torobo_rnn_utils import *
from rnn_state_publisher import *
import copy

# File paths
read_data = []
rnn_path = os.path.dirname(__file__) + "/mtrnn/"
out_file = rnn_path + "out/outValueClosed0"
out_steps = 399

def load_file(filepath, delimiter):
    """Load joint angles from a text file
       Save in array of joint angles in deg (format: [ LEFT-ARM RIGHT-ARM TORSO ])
       @filepath: path to file containing joint angles
       @delimiter: delimiter between joint angle values
    """
    fp = open(filepath, "r")
    for line in fp:
        read_data.append(line.split(delimiter))

def get_step_from_file(cur_pos, step):
    """Return scaled joint angles and velocities from loaded file
       Save in array of joint angles in deg (format: [ LEFT-ARM RIGHT-ARM TORSO ])
       @cur_pos: current position of joints
       @step: step index
    """
    tgt_pos = [float(read_data[step][i]) for i in range(ALL_JOINTS)]
    tgt_vel = [0.0 for i in range(ALL_JOINTS)]

    # Copy RNN input and output
    cur_pos_scaled = [rescale((cur_pos[i]-offset[i])/multiplier[i], scale_min[i], scale_max[i], -1.0, 1.0) for i in range(len(cur_pos))]
    rnn_input = copy.deepcopy(cur_pos_scaled)
    rnn_output = copy.deepcopy(tgt_pos)

    print "Unscaled output at step " + str(step)
    print (tgt_pos)
    
    for i in range(ALL_JOINTS):
        tgt_pos[i] = (rescale(tgt_pos[i], -1.0, 1.0, scale_min[i], scale_max[i]))*multiplier[i] + offset[i]
        # tgt_vel[i] = (cur_pos[i]-tgt_pos[i])/INTERVAL
        tgt_vel[i] = 0.0

    return tgt_pos, tgt_vel, rnn_input, rnn_output

def main():
    # Run once
    torobo = ToroboOperator()
    rnn_state_publisher = RnnStatePublisher(torobo)
    #move_homepos(torobo)
    load_file(out_file, " ")
    step = 0
    cur_pos, cur_vel = get_cur_joints(torobo)
    
    tgt_pos, tgt_vel, rnn_input, rnn_output = get_step_from_file(cur_pos, step)
    move_pos_time(torobo, tgt_pos, 3.0)
    rnn_state_publisher.publish(rnn_input, rnn_output)
    step += 1

    rate = rospy.Rate(1.0/INTERVAL)
    while not rospy.is_shutdown():
        cur_pos, cur_vel = get_cur_joints(torobo)
        tgt_pos, tgt_vel, rnn_input, rnn_output = get_step_from_file(cur_pos, step)
        move_pos(torobo, tgt_pos, tgt_vel)
        rnn_state_publisher.publish(rnn_input, rnn_output)
        step += 1
        if step >= out_steps:
            step = 0
        rate.sleep()

if __name__ == "__main__":
    main()

