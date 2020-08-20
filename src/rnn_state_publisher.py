#!/usr/bin/env python

import rospy
from torobo_operator import ToroboOperator
from sensor_msgs.msg import JointState

class RnnStatePublisher:
    def __init__(self, toroboOperator):
        self.torobo_ = toroboOperator
        self.rnn_state_ = self.init_rnn_joint_state(self.torobo_)
        self.rnn_state_pub_ = self.init_rnn_state_pub(self.torobo_)
    
    def init_rnn_joint_state(self, torobo):
        rnn_joint_state = JointState()
        joint_name = []
        joint_name.extend(torobo.get_joint_names(ToroboOperator.LEFT_ARM))
        joint_name.extend(torobo.get_joint_names(ToroboOperator.RIGHT_ARM))
        joint_name.extend(torobo.get_joint_names(ToroboOperator.TORSO_HEAD))

        for name in joint_name:
            rnn_joint_state.name.append("rnnin/"+name)
            rnn_joint_state.name.append("rnnout/"+name)
        rnn_joint_state.position = [0.0 for _ in range(len(rnn_joint_state.name))]
        rnn_joint_state.velocity = [0.0 for _ in range(len(rnn_joint_state.name))]

        return rnn_joint_state

    def init_rnn_state_pub(self, torobo):
        return rospy.Publisher('/torobo/rnn_state', JointState, queue_size=1)

    def publish(self, input_pos, output_pos, input_vel, output_vel):
        for i in range(len(input_pos)):
            idx = (i*2)
            self.rnn_state_.position[idx] = input_pos[i]
            self.rnn_state_.position[idx+1] = output_pos[i]
            self.rnn_state_.velocity[idx] = input_vel[i]
            self.rnn_state_.velocity[idx+1] = output_vel[i]
        self.rnn_state_.header.stamp = rospy.Time.now()
        self.rnn_state_pub_.publish(self.rnn_state_)
