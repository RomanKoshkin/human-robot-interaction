#!/usr/bin/env python

import os
import math
import rospy
from torobo_operator__upd import ToroboOperator

# Robot parameters 
INTERVAL = 0.02 # sec
REC_INTERVAL = 1.0/30.0 # Going too fast makes it hard to sync video to motor
CYCLIC_INTERVAL = rospy.Duration(secs=0, nsecs=INTERVAL*1000000000)
LIMIT_POS_DIFF = 20 # deg
LIMIT_VELOCITY = 20 # deg

ALL_JOINTS = 16 # not including grippers
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

def jointIDtoControllerID(j):
    if j >= 0 and j < LARM_JOINTS: # Left arm
        return ToroboOperator.LEFT_ARM, str(j+1)
    elif j >= LARM_JOINTS and j < LARM_JOINTS + RARM_JOINTS: # Right arm
        return ToroboOperator.RIGHT_ARM, str(j+1-LARM_JOINTS)
    elif j >= LARM_JOINTS + RARM_JOINTS and j < LARM_JOINTS + RARM_JOINTS + TOHD_JOINTS: # Torso head
        return ToroboOperator.TORSO_HEAD, str(j+1-(LARM_JOINTS + RARM_JOINTS))

def servo_on(torobo, left_arm=True, right_arm=True, torso_head=True):
    if left_arm:
        torobo.servo_on(ToroboOperator.LEFT_ARM, 'all')
        torobo.servo_on(ToroboOperator.LEFT_GRIPPER, 'all')
    if right_arm:
        torobo.servo_on(ToroboOperator.RIGHT_ARM, 'all')
        torobo.servo_on(ToroboOperator.RIGHT_GRIPPER, 'all')
    if torso_head:
        torobo.servo_on(ToroboOperator.TORSO_HEAD, 'all')

def servo_off(torobo, left_arm=True, right_arm=True, torso_head=True):
    if left_arm:
        torobo.servo_off(ToroboOperator.LEFT_ARM, 'all')
        torobo.servo_off(ToroboOperator.LEFT_GRIPPER, 'all')
    if right_arm:
        torobo.servo_off(ToroboOperator.RIGHT_ARM, 'all')
        torobo.servo_off(ToroboOperator.RIGHT_GRIPPER, 'all')
    if torso_head:
        torobo.servo_off(ToroboOperator.TORSO_HEAD, 'all')

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

def set_general_output_register(torobo, general_register_number, parameter_name):
    torobo.set_general_output_register(ToroboOperator.LEFT_ARM, general_register_number, parameter_name, ['all'])
    torobo.set_general_output_register(ToroboOperator.RIGHT_ARM, general_register_number, parameter_name, ['all'])
    torobo.set_general_output_register(ToroboOperator.TORSO_HEAD, general_register_number, parameter_name, ['all'])

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
    print "Moving to start position"
    set_position_control(torobo)
    torobo.move(ToroboOperator.LEFT_ARM, positions=[math.radians(90.0), math.radians(60.0), 0.0, math.radians(30.0), 0.0, 0.0], duration=5.0)
    torobo.move(ToroboOperator.RIGHT_ARM, positions=[math.radians(90.0), math.radians(60.0), 0.0, math.radians(30.0), 0.0, 0.0], duration=5.0)
    torobo.move(ToroboOperator.TORSO_HEAD, positions=[0.0, 0.0, 0.0, 0.0], duration=5.0)
    # torobo.move(ToroboOperator.LEFT_ARM, positions=[math.radians(100.0), math.radians(60.0), math.radians(50.0), math.radians(60.0), math.radians(60.0), math.radians(50.0)], duration=5.0)
    # torobo.move(ToroboOperator.RIGHT_ARM, positions=[math.radians(100.0), math.radians(60.0), math.radians(50.0), math.radians(60.0), math.radians(60.0), math.radians(50.0)], duration=5.0)
    # torobo.move(ToroboOperator.TORSO_HEAD, positions=[math.radians(50.0), math.radians(20.0), math.radians(40.0), math.radians(20.0)], duration=5.0)
    rospy.sleep(5)

def move_homepos(torobo):
    print "Moving to home position"
    set_position_control(torobo)
    torobo.move(ToroboOperator.LEFT_ARM, positions=[0.0, math.radians(90.0), 0.0, 0.0, 0.0, 0.0], duration=5.0)
    torobo.move(ToroboOperator.RIGHT_ARM, positions=[0.0, math.radians(90.0), 0.0, 0.0, 0.0, 0.0], duration=5.0)
    torobo.move(ToroboOperator.TORSO_HEAD, positions=[0.0, 0.0, 0.0, 0.0], duration=5.0)
    rospy.sleep(5)
    
def move_pos_time(torobo, pos, time):  
    print "Moving to position " + str(pos) + " in " + str(time) + "s"
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

def get_cur_gripper(torobo):
    lg_pos, lg_vel, lg_eft = torobo.get_joint_states(ToroboOperator.LEFT_GRIPPER)
    rg_pos, rg_vel, rg_eft = torobo.get_joint_states(ToroboOperator.RIGHT_GRIPPER)

    cur_pos = [lg_pos[0], rg_pos[0]]
    cur_vel = [lg_vel[0], rg_vel[0]]
    cur_eft = [lg_eft[0], rg_eft[0]]

    return array_degrees(cur_pos), array_degrees(cur_vel), cur_eft
