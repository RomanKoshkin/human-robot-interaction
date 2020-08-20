#!/usr/bin/env python

from torobo_rnn_utils import *
import sys
import select
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# For recording video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
rgb_avi = None
d_avi = None
bridge = CvBridge()
framew = 640
frameh = 480

savedir = "/home/torobo/workspace/recorded/"
t = 0
r = 0
rgb_flag = 0
d_flag = 0

def image_rgb_callback(msg):
    global rgb_flag
    global rgb_avi
    if rgb_flag > 0:
        try:
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError, e:
            print(e)
        else:
            #cv2.imwrite(savedir + "camera_rgb_frame.png", cv2_img)
            rgb_avi.write(cv2_img)
            rgb_flag -= 1

def image_d_callback(msg):
    global d_flag
    global d_avi
    if d_flag > 0:
        try:
            msg.encoding = "mono16" # Override encoding
            cv2_img = bridge.imgmsg_to_cv2(msg, "mono8")
        except CvBridgeError, e:
            print(e)
        else:
            #cv2.imwrite(savedir + "camera_d_frame.png", cv2_img)
            d_avi.write(cv2_img)
            d_flag -= 1

def main():
    global rgb_avi
    global rgb_flag
    global d_avi
    global d_flag
    torobo = ToroboOperator()
    
    read_data_suffix = "_motor"
    read_data = []
    delimiter = " "
    # Play accompanying video file
    rgb_suffix = "_camera_rgb.avi"
    d_suffix = "_camera_d.avi"
    cap_rgb = None
    cap_d = None
    play_video = False

    dub_mot_suffix = "x_motor"
    dub_eft_suffix = "x_torque"
    dub_record = False       # Record new file while playing back motion
    dub_rgb_suffix = "x_camera_rgb.avi"
    dub_d_suffix = "x_camera_d.avi"
    dub_record_video = False  # Record video as well (no preview)

    # Waiting for keyboard I/O
    timeout = 0.0001
    read_list = [sys.stdin]

    print "\n"
    t = int(raw_input("Trial number: ")) - 1
    r = int(raw_input("Run number: ")) - 1

    fp = open(savedir + str(t) + "_" + str(r) + read_data_suffix, "r")
    for line in fp:
        read_data.append(line.split(delimiter))
    fp.close()

    print "Read " + savedir + str(t) + "_" + str(r) + read_data_suffix + " (" + str(len(read_data)) + " lines)"

    if play_video:
        cap_rgb = cv2.VideoCapture(savedir + str(t) + "_" + str(r) + rgb_suffix)
        cap_d = cv2.VideoCapture(savedir + str(t) + "_" + str(r) + d_suffix)

        if not cap_rgb.isOpened():
            print "Failed to open video file " + savedir + str(t) + "_" + str(r) + rgb_suffix
        else:
            print "Opened video file " + savedir + str(t) + "_" + str(r) + rgb_suffix
        
        if not cap_d.isOpened():
            print "Failed to open video file " + savedir + str(t) + "_" + str(r) + d_suffix
        else:
            print "Opened video file " + savedir + str(t) + "_" + str(r) + d_suffix

    if dub_record:
        # Separate save files
        file_pos = savedir + str(t) + "_" + str(r) + dub_mot_suffix
        fpx = open(file_pos, "w")
        file_eft = savedir + str(t) + "_" + str(r) + dub_eft_suffix
        fex = open(file_eft, "w")

        if dub_record_video:
            image_rgb_topic = "/camera/color/image_raw"
            image_d_topic = "/camera/aligned_depth_to_color/image_raw"

            rgb_avi = cv2.VideoWriter(savedir + str(t) + "_" + str(r) + dub_rgb_suffix, fourcc, 1/REC_INTERVAL, (framew,frameh), True)
            d_avi = cv2.VideoWriter(savedir + str(t) + "_" + str(r) + dub_d_suffix, fourcc, 1/REC_INTERVAL, (framew,frameh), False)

            rospy.Subscriber(image_rgb_topic, Image, image_rgb_callback)
            rospy.Subscriber(image_d_topic, Image, image_d_callback)

    # Set velocity override here (0-100[%])
    set_velocity_override(torobo, 100.0)
    # Set softness override here (0-100[%])
    if dub_record:
        set_softness_override(torobo, 100.0) # Max soft at 100?
    else:
        set_softness_override(torobo, 50.0)

    if play_video:
        ret_rgb, frame_rgb = cap_rgb.read()
        ret_d, frame_d = cap_d.read()
        cv2.startWindowThread()

        if ret_rgb:
            cv2.namedWindow('RGB')
            cv2.moveWindow('RGB',0, 500)
            cv2.imshow('RGB', frame_rgb)
            cv2.waitKey(1)
        if ret_d:
            cv2.namedWindow('Depth')
            cv2.moveWindow('Depth', 800, 500)
            cv2.imshow('Depth', frame_d)
            cv2.waitKey(1)

    # !!!! Servo ON !!!!
    servo_on(torobo)
    rospy.sleep(1)

    # Move to initial position
    step = 0
    zero_vel = [0.0 for i in range(ALL_JOINTS)]
    tgt_pos = [float(read_data[step][i]) for i in range(ALL_JOINTS)]
    move_pos_time(torobo, tgt_pos, 5.0)
    step += 1

    # Set external force following online trajectory control mode
    set_external_force_following_online_trajectory_control(torobo)

    framerate = rospy.Rate(1/REC_INTERVAL)
    # Main loop
    while not rospy.is_shutdown():
        tgt_pos = [float(read_data[step][i]) for i in range(ALL_JOINTS)]
        print str(tgt_pos)
        move_pos_whole_body(torobo, tgt_pos, zero_vel)
        if play_video:
            ret_rgb, frame_rgb = cap_rgb.read()
            ret_d, frame_d = cap_d.read()

            if ret_rgb:
                cv2.imshow('RGB', frame_rgb)
                cv2.waitKey(1)
            if ret_d:
                cv2.imshow('Depth', frame_d)
                cv2.waitKey(1)

        if dub_record:
            if dub_record_video:
                # Attempt to synchronize video capture
                rgb_flag += 1
                d_flag += 1
            cur_pos, cur_vel = get_cur_joints(torobo)
            cur_eft = get_cur_joints_torque(torobo)

            for i in range(0, ALL_JOINTS):
                if i == 0:
                    str_pos = "%f" % cur_pos[i]
                    str_eft = "%f" % cur_eft[i]
                elif i == ALL_JOINTS-1:
                    str_pos += " %f\n" % cur_pos[i]
                    str_eft += " %f\n" % cur_eft[i]
                else:
                    str_pos += " %f" % cur_pos[i]
                    str_eft += " %f" % cur_eft[i]
            fpx.write(str_pos)
            fex.write(str_eft)
            print str_pos

        # Interval
        step += 1
        framerate.sleep()

        if select.select(read_list, [], [], timeout)[0]:
            print "Ending playback at step " + str(step)
            rospy.signal_shutdown("Playback aborted")

        if step >= len(read_data):
            print "Ending playback"
            rospy.signal_shutdown("Playback file ended")

    # Servo off
    rospy.sleep(2)
    servo_off(torobo)
    rospy.sleep(1)
    if dub_record:
        fpx.close()
        fex.close()
        if dub_record_video:
            rgb_avi.release()
            d_avi.release() 
    if play_video:
        cap_rgb.release()
        cap_d.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()