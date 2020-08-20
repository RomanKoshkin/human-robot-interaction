#! /usr/bin/python

# To import python codes from torobo_rnn
import sys
sys.path.insert(0, '/home/jungsik/catkin_ws/src/torobo_robot/torobo_rnn/scripts')
from torobo_rnn_utils import *

import os
import numpy as np
import rospy
import math

from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
# Instantiate CvBridge
bridge = CvBridge()



# The values are in degree, NOT RADIAN!
la_min = [70.0, 80.0, -30.0, 0.0]
la_max = [100.0, 90.0,  30.0, 15.0]

ra_min = la_min
ra_max = la_max

# Define the minimum and maximum of the head angles
head_min = [-20.0, -10.0]
head_max = [20.0,  20.0]

ARM_JNTS_FIXED = 0.0
TORSO_FIXED = 0.0

# The movement will be generated for 4 seconds
PLAY_DURATION = 4
# We'll capture 10 data points per second
CAPTURE_RATE = 10

NUM_TRIALS = 5

def main():
	# ==============================================================
	# Directory setting
	# ==============================================================
	isdir = os.path.exists("img-color")
	if not isdir:
		os.makedirs("img-color")
	isdir = os.path.exists("joints")
	if not isdir:
		os.makedirs("joints")

	# ==============================================================
	# Start the robot and initialize the settings
	# ==============================================================
	torobo = ToroboOperator()

	#image_topic = "/camera/color/image_raw" # Real-Robot
	image_topic = "/torobo/camera/color/image_raw"  # Simulation

	# Set velocity override here (0-100[%])
	set_velocity_override(torobo, 50.0)
	# Set softness override here (0-100[%])
	set_softness_override(torobo, 100.0)

	print "Setting the control mode..."
	set_position_control(torobo)
	
	print "Servo on"
	servo_on(torobo)
	rospy.sleep(5)

	print "Set to start position"
	move_startpos(torobo)
	rospy.sleep(5)

	print "Setting the control mode..."
	torobo.set_control_mode(ToroboOperator.TORSO_HEAD, 'all', 'position')
	torobo.set_control_mode(ToroboOperator.LEFT_ARM, 'all', 'external_force_following')
	torobo.set_control_mode(ToroboOperator.LEFT_ARM, '5', 'position')
	torobo.set_control_mode(ToroboOperator.LEFT_ARM, '6', 'position')
	torobo.set_control_mode(ToroboOperator.RIGHT_ARM, 'all', 'external_force_following')
	torobo.set_control_mode(ToroboOperator.RIGHT_ARM, '5', 'position')
	torobo.set_control_mode(ToroboOperator.RIGHT_ARM, '6', 'position')
	rospy.sleep(1)

	head_target_angle = [0.0, 20.0]

	for index_trial in xrange(NUM_TRIALS):
		print "Recording %d / %d " % (index_trial+1, NUM_TRIALS)
		f_rad = open("./joints/motor_rad_"+str(index_trial)+".txt", 'w')
		# ==============================================================
		# Head Random Babbling
		# ==============================================================
		head_target_angle[0] = np.random.uniform(head_min[0], head_max[0])
		head_target_angle[1] = np.random.uniform(head_min[1], head_max[1])

		torobo.move(ToroboOperator.TORSO_HEAD, positions=[TORSO_FIXED, TORSO_FIXED, math.radians(head_target_angle[0]), math.radians(head_target_angle[1])], duration=PLAY_DURATION)

		r = rospy.Rate(CAPTURE_RATE) # Hz
		index = 0
		while not rospy.is_shutdown():
			# ==============================================================
			# Receive JointState
			# ==============================================================
			cur_pos_deg, _ = get_cur_joints(torobo)
			cur_pos_rad = array_radians(cur_pos_deg)

			la_pos_rad = cur_pos_rad[0:LARM_JOINTS]
			ra_pos_rad = cur_pos_rad[LARM_JOINTS:LARM_JOINTS + RARM_JOINTS]
			torso_pos_rad = cur_pos_rad[LARM_JOINTS + RARM_JOINTS:LARM_JOINTS + RARM_JOINTS+2]
			head_pos_rad = cur_pos_rad[LARM_JOINTS + RARM_JOINTS+2:ALL_JOINTS]

			# Only first 4 joints are used in this code.
			for i in xrange(4):
				rad_data = "%.6f\t" % la_pos_rad[i]
				f_rad.write(rad_data)
			for i in xrange(4):
				rad_data = "%.6f\t" % ra_pos_rad[i]
				f_rad.write(rad_data)
			for i in xrange(2):
				rad_data = "%.6f\t" % head_pos_rad[i]
				f_rad.write(rad_data)
			f_rad.write("\n")
			
			# ==============================================================
			# Receive Camera Color Image
			# ==============================================================
			msg = rospy.wait_for_message(image_topic, Image)
			try:
				cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
			except CvBridgeError, e:
				print(e)
			else:
				cv2.imwrite('./img-color/camera_image_'+str(index_trial)+"_"+str(index)+'.jpeg', cv2_img)

			r.sleep()
			index = index + 1
			if(index > CAPTURE_RATE * PLAY_DURATION):
				break
		f_rad.close()
		now = rospy.get_rostime()

	print "Servo off"
	servo_off(torobo)
	rospy.sleep(5)


if __name__ == '__main__':
	main()

