#!/usr/bin/env python

from torobo_rnn_utils import *
import rospy

def main():
    torobo = ToroboOperator()

    # Set up the robot's motors
    set_external_force_following_online_trajectory_control(torobo)
    servo_on(torobo)
    rospy.sleep(2)
    move_homepos(torobo)
    # Print current joint angles
    pos,vel=get_cur_joints (torobo)
    print pos
    # Enter new joint angles
    # order: Left Arm J1-J6, Right Arm J1-J6, Torso J1-J4
    pos[0]=90
    pos[1]=45
    pos[2]=-90
    pos[3]=-45
    pos[14]=45
    pos[12]=-45
    move_pos_time(torobo,pos,3)
    pos[2]+=45
    move_pos_time(torobo,pos,2)
    pos[2]-=90
    move_pos_time(torobo,pos,3)
    pos[2]+=90
    move_pos_time(torobo,pos,3)
    
    # Move to new position in 5 seconds

    # Done!
    rospy.sleep(2)
    servo_off(torobo)
    rospy.sleep(1)

if __name__ == "__main__":
    main()
