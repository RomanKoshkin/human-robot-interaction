#! /usr/bin/python
# To import python codes from torobo_rnn
import sys
sys.path.insert(0, '/home/jungsik/catkin_ws/src/torobo_robot/torobo_rnn/scripts')
from torobo_rnn_utils import *

import rospy

CAPTURE_RATE = 10

def main():
	torobo = ToroboOperator()

	print "Setting the control mode..."
	torobo.set_control_mode(ToroboOperator.TORSO_HEAD, 'all', 'external_force_following')
	torobo.set_control_mode(ToroboOperator.LEFT_ARM, 'all', 'external_force_following')
	torobo.set_control_mode(ToroboOperator.LEFT_ARM, '5', 'position')
	torobo.set_control_mode(ToroboOperator.LEFT_ARM, '6', 'position')
	torobo.set_control_mode(ToroboOperator.RIGHT_ARM, 'all', 'external_force_following')
	torobo.set_control_mode(ToroboOperator.RIGHT_ARM, '5', 'position')
	torobo.set_control_mode(ToroboOperator.RIGHT_ARM, '6', 'position')

	print "Servo on"
	servo_on(torobo)
	rospy.sleep(5)
	now = rospy.get_rostime()

	r = rospy.Rate(CAPTURE_RATE) # Hz
	index = 0
	while not rospy.is_shutdown():
		# ==============================================================
		# Receive JointState
		# ==============================================================
		cur_pos_deg, _ = get_cur_joints(torobo)
		print "=" * 100
		print('L.ARM: \t %d \t %d \t %d \t %d' % (cur_pos_deg[0],cur_pos_deg[1],cur_pos_deg[2],cur_pos_deg[3]))
		print('R.ARM: \t %d \t %d \t %d \t %d' % (cur_pos_deg[6],cur_pos_deg[7],cur_pos_deg[8],cur_pos_deg[9]))
		print('Torso: \t %d \t %d' % (cur_pos_deg[12],cur_pos_deg[13]))
		print('Head: \t %d \t %d' % (cur_pos_deg[14],cur_pos_deg[15]))

		r.sleep()

	print "Servo off"
	servo_off(torobo)
	rospy.sleep(5)


if __name__ == '__main__':
	main()

