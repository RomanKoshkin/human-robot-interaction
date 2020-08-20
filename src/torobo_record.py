#!/usr/bin/env python

from torobo_rnn_utils import *
import numpy as np
import sys
import select
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# For video callback
rgb_prev = None
d_prev = None
bridge = CvBridge()

def main():
    global rgb_prev
    global d_prev
    torobo = ToroboOperator()

    ## Switches for recording
    rec_mot = True # Record joint angles (deg)
    print_mot = False # Print joint angles to console
    rec_grp = False # Record gripper joint angles (appended to mot)
    rec_eft = False # Record torque (N)
    rec_vel = False # Record joint velocity (deg/s)
    rec_rgb = True # Record RGB video
    rec_d = False # Record depth video (D435: IR emitter doesn't seem to turn on, image is very dark)
    d_masking = False # Mask RGB video using D (experimental, very short range)

    savedir = "/home/torobo/workspace/recorded/" # Directory to save in
    delimiter = " " # Delimiter used in motor output files

    # ROS topics for video input (requires CV bridge)
    image_rgb_topic = "/camera/color/image_raw"
    image_d_topic = "/camera/aligned_depth_to_color/image_raw"
  
    rec_mot_suffix = "_motor"
    rec_eft_suffix = "_torque"
    rec_vel_suffix = "_velocity"
    rec_rgb_suffix = "_camera_rgb.avi"
    rec_d_suffix = "_camera_d.avi"

    vid_framerate = 1.0/REC_INTERVAL # Update rate (too fast can result in dropped frames)
    framerate = rospy.Rate(vid_framerate)
    framew = 848
    frameh = 480
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

    ## Additional features
    # Use gripper torque sensors to actuate gripper
    gripper_ext_force = False # Rec gripper must also be true
    gripper_ext_force_debounce = 45 # Ignore torque input when actuation begins or ends
    gripper_ext_force_buffer = 3 # require a negative or positive hit for multiple frames
    gripper_ext_force_delta = 9.0 # Torque change required to actuate
    gripper_closed = 70.0 # How far to close (keep to a minimum)
    gripper_open = 1.0 # How far to open
    gripper_max_eft = 30.0 # How much force the gripper motor applies (keep to a minimum, but too low causes the gripper to be unresponsive)
    gripper_ext_force_hit = [0, 0]
    gripper_neutral_eft = [0.0, 0.0]
    gripper_active = [gripper_ext_force_debounce, gripper_ext_force_debounce]

    # Engage position control before hitting overposition (excludes grippers)
    soft_limiter = False
    soft_limiter_offset = 5.0 # Engage this much before hitting defined limit
    soft_limiter_active = [ False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False ]

    # Try to hold head in position despite body movement
    head_vhold = False
    head_vhold_offset = 0.0
    head_hhold = False
    head_vhold_offset = 0.0
    # Soft limit wrist movement
    wrist_hold = False
        
    # Waiting for keyboard I/O
    timeout = 0.0001
    read_list = [sys.stdin]

    t = 0
    r = 0
    mot_seq = [] if rec_mot else None
    eft_seq = [] if rec_eft else None
    vel_seq = [] if rec_vel else None
    rgb_seq = [] if rec_rgb else None 
    d_seq = [] if rec_d else None

    # Set velocity override here (0-100[%])
    #set_velocity_override(torobo, 100.0)
    # Set softness override here (0-100[%])
    #set_softness_override(torobo, 100.0) # Max soft at 100?

    # !!!! Servo ON !!!!
    # set_online_trajectory_control(torobo)
    # servo_on(torobo)
    # rospy.sleep(1)

    # Move to start position
    #move_startpos(torobo)
    #move_homepos(torobo)

    # Set teaching mode
    #set_external_force_following_control(torobo)
    print("Saving to " + savedir)
    try:
        t = int(raw_input("Trial number: "))
        r = int(raw_input("Run number: "))
    except ValueError:
        print("**Invalid number input, aborting")
        rospy.signal_shutdown("Recording aborted")
        exit(1)
    step = 0      

    # Engage position control on gripper
    if gripper_ext_force:
        print("Gripper position control for external force response")
        torobo.set_control_mode(ToroboOperator.LEFT_GRIPPER, '7', 'position')
        torobo.set_control_mode(ToroboOperator.RIGHT_GRIPPER, '7', 'position')

    # Try to maintain position on some joints
    if wrist_hold:
        print("Wrist trajectory control for hold")
        torobo.set_control_mode(ToroboOperator.LEFT_ARM, '6', 'external_force_following_online_trajectory')
        torobo.set_control_mode(ToroboOperator.RIGHT_ARM, '6', 'external_force_following_online_trajectory')

    if head_vhold:
        print("Neck trajectory control for hold (vertical)")
        torobo.set_control_mode(ToroboOperator.TORSO_HEAD, '4', 'external_force_following_online_trajectory')
        cur_pos, _ = get_cur_joints(torobo)
        head_vhold_offset = cur_pos[15]

    if head_hhold:
        print("Neck trajectory control for hold (horizontal)")
        torobo.set_control_mode(ToroboOperator.TORSO_HEAD, '3', 'external_force_following_online_trajectory')
        cur_pos, _ = get_cur_joints(torobo)
        head_hhold_offset = cur_pos[14]

    if rec_rgb or rec_d:
        cv2.startWindowThread()
        if rec_rgb:    
            rospy.Subscriber(image_rgb_topic, Image, image_rgb_callback)
        if rec_d:
            rospy.Subscriber(image_d_topic, Image, image_d_callback)

    raw_input("Move to starting position, then press Enter to begin recording")

    if rec_rgb:
        cv2.namedWindow('RGB')
        cv2.moveWindow('RGB',0, 500)
    
    if rec_d:
        cv2.namedWindow('Depth')
        cv2.moveWindow('Depth', 800, 500)

    print("Recording (press Enter to stop)...")
    # Main loop
    while not rospy.is_shutdown():
        cur_pos, cur_vel = get_cur_joints(torobo)
        cur_eft = get_cur_joints_torque(torobo)
        if rec_grp or gripper_ext_force:
            cur_gripper_pos, cur_gripper_vel, cur_gripper_eft = get_cur_gripper(torobo)

        # Store joint angles
        if rec_mot:
            pos = cur_pos
            if rec_grp:
                pos += cur_gripper_pos
            store_motor(mot_seq, pos, delimiter, print_mot)
        
        if rec_eft:
            eft = cur_eft
            if rec_grp:
                eft += cur_gripper_eft
            store_motor(eft_seq, eft, delimiter, False)

        if rec_vel:
            vel = cur_vel
            if rec_grp:
                vel += cur_gripper_vel
            store_motor(vel_seq, vel, delimiter, False)

        # Store video frames
        if rec_rgb or rec_d:
            store_video_frame(rgb_seq, d_seq)

        # Attempt to show last video frame
        if rgb_prev is not None:
            if d_masking is True and d_prev is not None:
                prev = rgb_prev * np.expand_dims(d_prev, axis=2)
            else:
                prev = rgb_prev
            cv2.imshow('RGB', prev)
            cv2.waitKey(1)
        if d_prev is not None:
            cv2.imshow('Depth', d_prev)
            cv2.waitKey(1)

        ## Additional features
        # Check for gripper activation
        if gripper_ext_force:
            for i in range(len(cur_gripper_eft)):
                if gripper_active[i] == 0:
                    gripperDeltaEft = cur_gripper_eft[i] - gripper_neutral_eft[i]
                    if abs(gripperDeltaEft) > gripper_ext_force_delta: # hit
                        # print("hit " + str(gripperDeltaEft))
                        gripper_ext_force_hit[i] += 1
                        if abs(gripper_ext_force_hit[i]) > gripper_ext_force_buffer:
                            if gripperDeltaEft < 0.0: # close
                                gripper_pos = np.radians(gripper_closed)
                                # print("close!")
                            elif gripperDeltaEft > 0.0: # open
                                gripper_pos = np.radians(gripper_open)
                                # print("open!")
                            if i == 0:
                                # print("move left gripper")
                                torobo.move_gripper(ToroboOperator.LEFT_GRIPPER, position=gripper_pos, max_effort=gripper_max_eft)
                            elif i == 1:
                                # print("move right gripper")
                                torobo.move_gripper(ToroboOperator.RIGHT_GRIPPER, position=gripper_pos, max_effort=gripper_max_eft)
                            gripper_active[i] = gripper_ext_force_debounce
                    else:
                        gripper_ext_force_hit[i] = 0 # reset
                else: # gripper is moving
                    gripper_active[i] -= 1
                    gripperDeltaEft = cur_gripper_eft[i] - gripper_neutral_eft[i]
                    if gripper_active[i] == 0:
                        if abs(gripperDeltaEft) > gripper_ext_force_delta-2.5: # debounced but torque is still on
                            # print("continued hit " + str(gripperDeltaEft))
                            gripper_ext_force_hit[i] += 1
                            if i == 0:
                                # print("relax left gripper")
                                torobo.set_control_mode(ToroboOperator.LEFT_GRIPPER, '7', 'current')
                                rospy.sleep(0.5)
                                torobo.set_control_mode(ToroboOperator.LEFT_GRIPPER, '7', 'position')
                                
                            elif i == 1:
                                # print("relax right gripper")
                                torobo.set_control_mode(ToroboOperator.RIGHT_GRIPPER, '7', 'current')
                                rospy.sleep(0.5)
                                torobo.set_control_mode(ToroboOperator.RIGHT_GRIPPER, '7', 'position')
                            # gripper_active[i] += 10
                        # else:
                        #     gripper_ext_force_hit[i] = 0
                        elif gripper_ext_force_hit[i] > 0:
                            gripper_active[i] += 10
                            gripper_ext_force_hit[i] -= 1
                            # print("waiting " + str(gripper_ext_force_hit[i]))

        # Resist moving to overposition
        if soft_limiter:
            for i in range(0, ALL_JOINTS):
                if i == 15 and head_vhold:
                    continue
                if i == 14 and head_hhold:
                    continue
                if cur_pos[i] > scale_max[i]-soft_limiter_offset or cur_pos[i] < scale_min[i]+soft_limiter_offset:
                    c, j = jointIDtoControllerID(i)
                    torobo.set_control_mode(c, j, 'external_force_following_online_trajectory')
                    soft_limiter_active[i] = True
                    print("**Soft limiter active on J" + str(i+1))
                else:
                    if soft_limiter_active[i]:
                        c, j = jointIDtoControllerID(i)
                        torobo.set_control_mode(c, j, 'external_force_following')
                        soft_limiter_active[i] = False

        # Hold head position
        if head_vhold or head_hhold:
            torsoJ4 = -cur_pos[13] + head_vhold_offset
            if not (head_vhold and torsoJ4 > scale_min[15]+soft_limiter_offset and torsoJ4 < scale_max[15]-soft_limiter_offset):
                torsoJ4 = cur_pos[15]
            torsoJ3 = -cur_pos[12] + head_hhold_offset
            if not (head_hhold and torsoJ3 > scale_min[14]+soft_limiter_offset and torsoJ3 < scale_max[14]-soft_limiter_offset):
                torsoJ3 = cur_pos[14]

            torobo.move(ToroboOperator.TORSO_HEAD, positions=[math.radians(cur_pos[12]), math.radians(cur_pos[13]), math.radians(torsoJ3), math.radians(torsoJ4)], duration=INTERVAL)

        # Interval
        step += 1
        framerate.sleep()

        # Key pressed
        if select.select(read_list, [], [], timeout)[0]:
            # Save buffer to files
            print("Saving...")
            if rec_mot:
                save_motor(mot_seq, savedir + str(t) + "_" + str(r) + rec_mot_suffix)
            if rec_eft:
                save_motor(eft_seq, savedir + str(t) + "_" + str(r) + rec_eft_suffix)
            if rec_vel:
                save_motor(vel_seq, savedir + str(t) + "_" + str(r) + rec_vel_suffix)
            
            if rec_rgb:
                save_video(rgb_seq, savedir + str(t) + "_" + str(r) + rec_rgb_suffix, fourcc, vid_framerate, (framew, frameh))
            if rec_d:
                save_video(d_seq, savedir + str(t) + "_" + str(r) + rec_d_suffix, fourcc, vid_framerate, (framew, frameh))
            print("Recording done (" + str(step) + " frames)")
            # print "Ending recording with " + str(step) + " steps (RGB dropped frames " + str(rgb_flag) + ", D dropped frames " + str(d_flag) + ")"
            rospy.signal_shutdown("Recording done")

    if rec_rgb or rec_d:
        cv2.destroyAllWindows()

    # Servo off
    # rospy.sleep(2)
    # servo_off(torobo)
    # rospy.sleep(1)

def image_rgb_callback(msg):
    global rgb_prev
    global bridge
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        rgb_prev = cv2_img

def image_d_callback(msg):
    global d_prev
    global bridge
    try:
        # Source is 16 bit but display as 8 bit color
        msg.encoding = "mono16"
        cv2_img = bridge.imgmsg_to_cv2(msg, "mono8")
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
    except CvBridgeError, e:
        print(e)
    else:
        d_prev = cv2_img

# Save in memory (video)
def store_video_frame(rgb_seq, d_seq):
    global rgb_prev
    global d_prev
    if rgb_seq is not None:
        rgb_seq.append(rgb_prev)
    if d_seq is not None:
        d_seq.append(d_prev)

def save_video(vid, file_vid, fourcc, avi_framerate, framesize, color=True):
    try:
        avi = cv2.VideoWriter(file_vid, fourcc, avi_framerate, (framesize[0], framesize[1]), isColor=color)
        for i in range(0, len(vid)):
            avi.write(vid[i])
        avi.release()
    except cv2.error as e:
        print("**Failed to save video!")

# Save in memory (motor)
def store_motor(mot_seq, joints, delimiter, print_mot=False):
    for i in range(0, len(joints)):
        if i == 0:
            mot_str = "%f" % joints[i]
        elif i == len(joints)-1:
            mot_str += delimiter + "%f\n" % joints[i]
        else:
            mot_str += delimiter + "%f" % joints[i]
    mot_seq.append(mot_str)
    if print_mot:
        print(mot_str)

def save_motor(mot_seq, file_rec):
    fp = open(file_rec, "w")
    for i in range(0, len(mot_seq)):
        fp.write(mot_seq[i])
    fp.close()

if __name__ == "__main__":
    main()
