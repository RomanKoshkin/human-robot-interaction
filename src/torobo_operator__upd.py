#!/usr/bin/env python

import time
import math
import rospy
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from torobo_msgs.msg import ToroboDynamics 

from sensor_msgs.msg import JointState
from control_msgs.msg import GripperCommand, GripperCommandActionGoal
import torobo_collision_detector.check_collision_client
from torobo_driver import servo_off_client
from torobo_driver import servo_on_client
from torobo_driver import set_control_mode_client
from torobo_driver import set_robot_controller_parameter_client
from torobo_driver import set_general_output_register_client

 # /torobo/right_arm_controller/torobo_dynamics


class ToroboOperator:

    __CONTROLLER_ID = (
        LEFT_ARM,
        RIGHT_ARM,
        TORSO_HEAD,
        LEFT_GRIPPER,
        RIGHT_GRIPPER,
    ) = range(0, 5)

    __CONTROLLER_NAME_LIST = {
        LEFT_ARM     : 'left_arm_controller',
        RIGHT_ARM    : 'right_arm_controller',
        TORSO_HEAD   : 'torso_head_controller',
        LEFT_GRIPPER : 'left_gripper_controller',
        RIGHT_GRIPPER: 'right_gripper_controller',
    }

    __JOINT_NAME_LIST = {
        LEFT_ARM     : ['left_arm/joint_1', 'left_arm/joint_2', 'left_arm/joint_3', 'left_arm/joint_4', 'left_arm/joint_5', 'left_arm/joint_6', ],
        RIGHT_ARM    : ['right_arm/joint_1', 'right_arm/joint_2', 'right_arm/joint_3', 'right_arm/joint_4', 'right_arm/joint_5', 'right_arm/joint_6', ],
        TORSO_HEAD   : ['torso_head/joint_1', 'torso_head/joint_2', 'torso_head/joint_3', 'torso_head/joint_4', ],
        LEFT_GRIPPER : ['left_gripper/finger_joint', ],
        RIGHT_GRIPPER: ['right_gripper/finger_joint', ],
    }


    def __rostopic_exists(self, topic_name):
        """Checking exist of rostopic.
        """
        topic_list = rospy.get_published_topics()
        for topic in topic_list:
            if topic[0] == topic_name:
                return True
        return False


    def __init__(self):

        # Init rospy
        rospy.init_node('torobo_operator_node', anonymous=True)

        # check topic type of gripper
        if self.__rostopic_exists('/torobo/left_gripper_controller/command'):
            self.__isSim=False
        else:
            self.__isSim=True

        if self.__isSim:
            self.__PUBLISHER_LIST = {
                ToroboOperator.LEFT_ARM     : rospy.Publisher('/torobo/left_arm_controller/command', JointTrajectory, queue_size=1),
                ToroboOperator.RIGHT_ARM    : rospy.Publisher('/torobo/right_arm_controller/command', JointTrajectory, queue_size=1),
                ToroboOperator.TORSO_HEAD   : rospy.Publisher('/torobo/torso_head_controller/command', JointTrajectory, queue_size=1),
                ToroboOperator.LEFT_GRIPPER : rospy.Publisher('/torobo/left_gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=1),
                ToroboOperator.RIGHT_GRIPPER: rospy.Publisher('/torobo/right_gripper_controller/gripper_cmd/goal', GripperCommandActionGoal, queue_size=1),
            }
        else:
            self.__PUBLISHER_LIST = {
                ToroboOperator.LEFT_ARM     : rospy.Publisher('/torobo/left_arm_controller/command', JointTrajectory, queue_size=1),
                ToroboOperator.RIGHT_ARM    : rospy.Publisher('/torobo/right_arm_controller/command', JointTrajectory, queue_size=1),
                ToroboOperator.TORSO_HEAD   : rospy.Publisher('/torobo/torso_head_controller/command', JointTrajectory, queue_size=1),
                ToroboOperator.LEFT_GRIPPER : rospy.Publisher('/torobo/left_gripper_controller/command', GripperCommand, queue_size=1),
                ToroboOperator.RIGHT_GRIPPER: rospy.Publisher('/torobo/right_gripper_controller/command', GripperCommand, queue_size=1),
            }

        # Create service client
        self.__servo_off_client = {}
        self.__servo_on_client = {}
        self.__set_control_mode_client = {}
        self.__set_robot_controller_parameter_client = {}
        for (controller_id, controller_name) in ToroboOperator.__CONTROLLER_NAME_LIST.items():
            self.__servo_off_client[controller_name] = servo_off_client.ServoOffClient('/torobo/' + controller_name)
            self.__servo_on_client[controller_name] = servo_on_client.ServoOnClient('/torobo/' + controller_name)
            self.__set_control_mode_client[controller_name] = set_control_mode_client.SetControlModeClient('/torobo/' + controller_name)
            self.__set_robot_controller_parameter_client[controller_name] = set_robot_controller_parameter_client.SetRobotControllerParameterClient('/torobo/' + controller_name)

        # current joint_states
        self.__current_position = {}
        self.__current_velocity = {}
        self.__current_effort   = {}

        # torobo dynamics 
        self.__gravity_compensation_effort   = {}
        self.__ref_dynamics_effort   = {}
        self.__cur_dynamics_effort   = {}
        self.__inertia_diagonal   = {}

        # Create subscriber
        rospy.Subscriber('/torobo/joint_states', JointState, self.__callback, queue_size=1, tcp_nodelay=True)

        # Create subscriber Robot Dynamics
        rospy.Subscriber('/torobo/torso_head_controller/torobo_dynamics', ToroboDynamics, self.__callback_Dynamics, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/torobo/right_arm_controller/torobo_dynamics', ToroboDynamics, self.__callback_Dynamics, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/torobo/left_arm_controller/torobo_dynamics', ToroboDynamics, self.__callback_Dynamics, queue_size=1, tcp_nodelay=True)

        # Before publishing message, you need to wait a moment
        #   (around one second) until publisher has been initialized.
        #   Otherwise, published message will be dropped by the publisher.
        rospy.sleep(2)


    def __callback(self, joint_state):
        """Callback for subscriber of joint_states.
        """
        # update current joint_states.
        for name, position, velocity, effort in zip(joint_state.name, joint_state.position, joint_state.velocity, joint_state.effort):
            self.__current_position[name] = position
            self.__current_velocity[name] = velocity
            self.__current_effort  [name] = effort

    def __callback_Dynamics(self, dynamics):
        """Callback for subscriber of dynamics.
        """    
        for name, gravity_compensation_effort, ref_dynamics_effort, cur_dynamics_effort, inertia_diagonal in zip(dynamics.name, dynamics.gravity_compensation_effort, dynamics.ref_dynamics_effort, dynamics.cur_dynamics_effort,  dynamics.inertia_diagonal):

            self.__gravity_compensation_effort[name] = gravity_compensation_effort
            self.__ref_dynamics_effort[name] =  ref_dynamics_effort 
            self.__cur_dynamics_effort[name] = cur_dynamics_effort
            self.__inertia_diagonal[name]  = inertia_diagonal
                        
    def __check_collision(self, names, positions, velocities):
        """Checking collision
            @names     : joint_name
            @positions : [radian](arm) or [meter](gripper)
            @velocities: [radian/sec](arm) or [meter/sec](gripper)
        """
        joint_state = JointState()
        joint_state.name = names
        joint_state.position = positions
        joint_state.velocity = velocities
        joint_state.effort = [0.0 for _ in range(len(names))]
        ret = torobo_collision_detector.check_collision_client.call_service(rospy.get_namespace(), joint_state)
        return ret


    def __create_joint_trajectory(self, joint_names, positions, velocities, accelerations, duration):
        """Creating JointTrajectory message.
            @joint_names  : joint_name
            @positions    : [radian]
            @velocities   : [radian/sec]
            @accelerations: [radian/sec^2]
            @duration     : [sec]
        """
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ''
        msg.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = velocities
        point.accelerations = accelerations
        point.effort = [0.0 for _ in range(len(joint_names))]
        point.time_from_start = rospy.Duration(secs=duration, nsecs=0)
        msg.points.append(point)
        return msg


    def __create_gripper_command(self, position, max_effort):
        """Creating GripperCommand message.
            @position   : [m]
            @max_effort : [N]
        """
        if self.__isSim:
            msg = GripperCommandActionGoal()
            msg.goal.command.position   = position
            msg.goal.command.max_effort = max_effort
        else:
            msg = GripperCommand()
            msg.position   = position
            msg.max_effort = max_effort
        return msg

    def get_joint_names(self, controller_id):
        """You can get joint_names of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM', 'TORSO_HEAD', 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
        """
        return ToroboOperator.__JOINT_NAME_LIST[controller_id]

    def get_joint_states(self, controller_id):
        """You can get current joint_states (containing positions, velocities and efforts) of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM', 'TORSO_HEAD', 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
        """
        position = []
        velocity = []
        effort   = []
        for joint_name in ToroboOperator.__JOINT_NAME_LIST[controller_id]:
            position.append(self.__current_position[joint_name])
            velocity.append(self.__current_velocity[joint_name])
            effort  .append(self.__current_effort  [joint_name])
        return position, velocity, effort

    def get_robot_dynamics(self, controller_id):
        """You can get current dynamics (gravity_compensation_effort, ref_dynamics_effort, cur_dynamics_effort, inertia_diagonal) of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM', 'TORSO_HEAD', 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
        """
        gravity_compensation_effort = []
        ref_dynamics_effort = []
        cur_dynamics_effort = []
        inertia_diagonal = []

        for joint_name in ToroboOperator.__JOINT_NAME_LIST[controller_id]:
            gravity_compensation_effort.append(self.__gravity_compensation_effort[joint_name])
            ref_dynamics_effort.append(self.__ref_dynamics_effort[joint_name])
            cur_dynamics_effort.append(self.__cur_dynamics_effort[joint_name])
            inertia_diagonal.append(self.__inertia_diagonal[joint_name])
            
        return gravity_compensation_effort, ref_dynamics_effort, cur_dynamics_effort, inertia_diagonal

    def servo_on(self, controller_id, joint_names):
        """You can set servo on of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM', 'TORSO_HEAD', 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
            @joint_names : target joint names list
        """
        controller_name = ToroboOperator.__CONTROLLER_NAME_LIST[controller_id]
        self.__servo_on_client[controller_name].call_service(joint_names)

    def servo_off(self, controller_id, joint_names):
        """You can set servo off of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM', 'TORSO_HEAD', 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
            @joint_names : target joint names list
        """
        controller_name = ToroboOperator.__CONTROLLER_NAME_LIST[controller_id]
        self.__servo_off_client[controller_name].call_service(joint_names)

    def set_control_mode(self, controller_id, joint_names, control_mode):
        """You can set joint control mode of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM', 'TORSO_HEAD', 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
            @control_mode : 'position', 'external_force_following', 'online_trajectory' or 'external_force_following_online_trajectory'.
        """
        controller_name = ToroboOperator.__CONTROLLER_NAME_LIST[controller_id]
        self.__set_control_mode_client[controller_name].call_service(joint_names, control_mode)

    def set_robot_controller_parameter(self, controller_id, parameter_name, parameter_values, joint_names):
        """You can set robot controller paramter of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM', 'TORSO_HEAD', 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
            @paramter_name : 'velocity_override' or 'softness_override', etc... (There are defined in torbo_driver/torobo_easy_command.py [paramNameNumberDict_])
            @paramter_values : paramter values list
            @joint_names : target joint names list
        """
        controller_name = ToroboOperator.__CONTROLLER_NAME_LIST[controller_id]
        self.__set_robot_controller_parameter_client[controller_name].call_service(parameter_name, parameter_values, joint_names)

    def set_general_output_register(self, controller_id, general_register_number, parameter_name, joint_names):
        """You can set general output register of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM', 'TORSO_HEAD', 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
            @general_register_number : target general output register number [0,1,2,3]
            @paramter_name : 'velocity_override' or 'softness_override', etc... (There are defined in torbo_driver/torobo_easy_command.py [paramNameNumberDict_])
            @joint_names : target joint names list
        """
        controller_name = ToroboOperator.__CONTROLLER_NAME_LIST[controller_id]
        self.__set_general_output_register_client[controller_name].call_service(general_register_number, parameter_name, joint_names)

    def move(self, controller_id, positions, velocities=None, accelerations=None, duration=5.0):
        """You can move an arm of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM' or 'TORSO_HEAD'.
            @positions     : target position     [rad]
            @velocities    : target velocity     [rad/sec]   (optional. default is 0.0)
            @accelerations : target acceleration [rad/sec^2] (optional. default is 0.0)
            @duration      : duration time       [sec]       (optional. default is 5.0)
        """

        joint_names   = ToroboOperator.__JOINT_NAME_LIST[controller_id]
        velocities    = [0.0 for _ in range(len(joint_names))] if velocities    is None else velocities
        accelerations = [0.0 for _ in range(len(joint_names))] if accelerations is None else accelerations
        if not (len(joint_names) == len(positions) == len(velocities) == len(accelerations)):
            rospy.logerr('[move_arm] invalid arguments')
        
        isCollided = self.__check_collision(joint_names, positions, velocities)
        if isCollided:
            rospy.logwarn('[move_arm] trajectory is canceled due to collision')
            return

        msg = self.__create_joint_trajectory(joint_names, positions, velocities, accelerations, duration)
        self.__PUBLISHER_LIST[controller_id].publish(msg)

    def move_joint_my(self, controller_id, joint_ids, positions, velocities=None, accelerations=None, duration=5.0):
        """You can move a specified joint (joints) on an effector of specified controller.
            @controller_id : 'LEFT_ARM', 'RIGHT_ARM' or 'TORSO_HEAD'.
            @positions     : target position     [rad]
            @velocities    : target velocity     [rad/sec]   (optional. default is 0.0)
            @accelerations : target acceleration [rad/sec^2] (optional. default is 0.0)
            @duration      : duration time       [sec]       (optional. default is 5.0)
        """
        # get joint names, set velocities and accelerations:
        joint_names   = [ToroboOperator.__JOINT_NAME_LIST[controller_id][i] for i in joint_ids]
        velocities    = [0.0 for _ in range(len(joint_names))] if velocities    is None else velocities
        accelerations = [0.0 for _ in range(len(joint_names))] if accelerations is None else accelerations
        
        if not (len(joint_names) == len(positions) == len(velocities) == len(accelerations)):
            rospy.logerr('[move_arm] invalid arguments')
        
        isCollided = self.__check_collision(joint_names, positions, velocities)
        if isCollided:
            rospy.logwarn('[move_arm] trajectory is canceled due to collision')
            return
        
        msg = self.__create_joint_trajectory(joint_names, positions, velocities, accelerations, duration)
        self.__PUBLISHER_LIST[controller_id].publish(msg)

    def move_joints(self, 
                    left_arm_joint_names=None, left_arm_positions=None, left_arm_velocities=None, left_arm_accelerations=None,
                    right_arm_joint_names=None, right_arm_positions=None, right_arm_velocities=None, right_arm_accelerations=None,
                    torso_head_joint_names=None, torso_head_positions=None, torso_head_velocities=None, torso_head_accelerations=None,
                    duration=5.0):
        """You can move specified joints.
            @joint_names   : target joint name
            @positions     : target position     [rad]
            @velocities    : target velocity     [rad/sec]   (optional. default is 0.0)
            @accelerations : target acceleration [rad/sec^2] (optional. default is 0.0)
            @duration      : duration time       [sec]       (optional. default is 5.0)
        """
        controller_ids = []
        msgs = []
        joint_names = []
        positions = []
        velocities = []
        accelerations = []
        if((left_arm_joint_names is not None) and (len(left_arm_joint_names) == len(left_arm_positions))):
            controller_ids.append(ToroboOperator.LEFT_ARM)
            joint_names.extend(left_arm_joint_names)
            positions.extend(left_arm_positions)
            left_arm_velocities    = [0.0 for _ in range(len(left_arm_joint_names))] if left_arm_velocities       is None else left_arm_velocities
            left_arm_accelerations = [0.0 for _ in range(len(left_arm_joint_names))] if left_arm_accelerations    is None else left_arm_accelerations
            velocities.extend(left_arm_velocities)
            accelerations.extend(left_arm_accelerations)
            msgs.append(self.__create_joint_trajectory(left_arm_joint_names, left_arm_positions, left_arm_velocities, left_arm_accelerations, duration))
        if((right_arm_joint_names is not None) and (len(right_arm_joint_names) == len(right_arm_positions))):
            controller_ids.append(ToroboOperator.RIGHT_ARM)
            joint_names.extend(right_arm_joint_names)
            positions.extend(right_arm_positions)
            right_arm_velocities    = [0.0 for _ in range(len(right_arm_joint_names))] if right_arm_velocities       is None else right_arm_velocities
            right_arm_accelerations = [0.0 for _ in range(len(right_arm_joint_names))] if right_arm_accelerations    is None else right_arm_accelerations
            velocities.extend(right_arm_velocities)
            accelerations.extend(right_arm_accelerations)
            msgs.append(self.__create_joint_trajectory(right_arm_joint_names, right_arm_positions, right_arm_velocities, right_arm_accelerations, duration))
        if((torso_head_joint_names is not None) and (len(torso_head_joint_names) == len(torso_head_positions))):
            controller_ids.append(ToroboOperator.TORSO_HEAD)
            joint_names.extend(torso_head_joint_names)
            positions.extend(torso_head_positions)
            torso_head_velocities    = [0.0 for _ in range(len(torso_head_joint_names))] if torso_head_velocities       is None else torso_head_velocities
            torso_head_accelerations = [0.0 for _ in range(len(torso_head_joint_names))] if torso_head_accelerations    is None else torso_head_accelerations
            velocities.extend(torso_head_velocities)
            accelerations.extend(torso_head_accelerations)
            msgs.append(self.__create_joint_trajectory(torso_head_joint_names, torso_head_positions, torso_head_velocities, torso_head_accelerations, duration))

        if not ((controller_ids != []) or (len(joint_names) == len(positions) == len(velocities) == len(accelerations))):
            rospy.logerr('[move_arm] invalid arguments')

        isCollided = self.__check_collision(joint_names, positions, velocities)
        if isCollided:
            rospy.logwarn('[move_arm] trajectory is canceled due to collision')
            return

        for(controller_id, msg) in zip(controller_ids, msgs):
            self.__PUBLISHER_LIST[controller_id].publish(msg)


    def move_gripper(self, controller_id, position, max_effort=20.0):
        """You can move a gripper of specified controller.
            @controller_id : 'LEFT_GRIPPER' or 'RIGHT_GRIPPER'.
            @position      : target position     [m]
            @max_effort    : max_effort          [N]         (optional, default is 20.0)
        """
        msg = self.__create_gripper_command(position, max_effort)
        self.__PUBLISHER_LIST[controller_id].publish(msg)

def main():
    torobo = ToroboOperator()

    print "moving home"
    torobo.move(ToroboOperator.LEFT_ARM, positions=[0.0, math.radians(90.0), 0.0, 0.0, 0.0, 0.0], duration=5.0)
    torobo.move(ToroboOperator.RIGHT_ARM, positions=[0.0, math.radians(90.0), 0.0, 0.0, 0.0, 0.0], duration=5.0)
    torobo.move(ToroboOperator.TORSO_HEAD, positions=[0.0, 0.0, 0.0, 0.0], duration=5.0)
    rospy.sleep(5)

    print "moving to target"
    torobo.move(ToroboOperator.LEFT_ARM, positions=[math.radians(60.0), math.radians(90.0), math.radians(90.0), math.radians(-30.0), 0.0, math.radians(30.0)], duration=5.0)
    torobo.move(ToroboOperator.RIGHT_ARM, positions=[math.radians(90.0), math.radians(60.0), 0.0, math.radians(-30.0), 0.0, math.radians(30.0)], duration=5.0)
    torobo.move(ToroboOperator.TORSO_HEAD, positions=[0.0, math.radians(30.0), 0.0, math.radians(40.0)], duration=5.0)
    rospy.sleep(5)

    print "gripper open"
    torobo.move_gripper(ToroboOperator.LEFT_GRIPPER, position=0.08)
    torobo.move_gripper(ToroboOperator.RIGHT_GRIPPER, position=0.08)
    rospy.sleep(2)

    print "gripper close"
    torobo.move_gripper(ToroboOperator.LEFT_GRIPPER, position=0.0)
    torobo.move_gripper(ToroboOperator.RIGHT_GRIPPER, position=0.0)
    rospy.sleep(2)


if __name__ == "__main__":
    main()

