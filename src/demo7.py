#!/usr/bin/env python

import rospy
import cv2
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import cv_bridge
from geometry_msgs.msg import Twist, Pose, PoseStamped, PointStamped
from smach import State, StateMachine
import smach_ros
from dynamic_reconfigure.server import Server
from nav_msgs.msg import Odometry
from ar_track_alvar_msgs.msg import AlvarMarkers
from kobuki_msgs.msg import BumperEvent, Sound, Led
from tf.transformations import decompose_matrix, compose_matrix
from ros_numpy import numpify
from sensor_msgs.msg import Joy, LaserScan, Image
import numpy as np
import angles as angles_lib
import math
import random
from std_msgs.msg import Bool, String, Int32
from ar_track_alvar_msgs.msg import AlvarMarkers
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf2_ros
import tf2_geometry_msgs
import tf

TAGS_FOUND = []
START_POSE = None
TAG_POSE = None
CURRENT_POSE = None
client = None
TAGS_IN_TOTAL = 3
CURRENT_STATE = None

PUSH_GOAL = None
END_GOAL = None
MARKER_POSE = None
START = True


class Turn(State):
    """
    Turning a specific angle, based on Sean's example code from demo2
    """

    def __init__(self, angle=90):
        State.__init__(self, outcomes=["done"])
        self.tb_position = None
        self.tb_rot = None
        # angle defines angle target relative to goal direction
        self.angle = angle

        # pub / sub
        rospy.Subscriber("odom", Odometry, callback=self.odom_callback)
        self.cmd_pub = rospy.Publisher(
            "cmd_vel", Twist, queue_size=1)

    def odom_callback(self, msg):
        tb_pose = msg.pose.pose
        __, __, angles, position, __ = decompose_matrix(numpify(tb_pose))
        self.tb_position = position[0:2]
        self.tb_rot = angles

    def execute(self, userdata):
        turn_direction = 1

        start_pose = [0, 0, 0, 0]
        if self.angle == 0:  # target is goal + 0
            goal = start_pose[1]
        elif self.angle == 90:  # target is goal + turn_direction * 90
            goal = start_pose[1] + np.pi/2 * turn_direction
        elif self.angle == 180:  # target is goal + turn_direction * 180
            goal = start_pose[1] + np.pi * turn_direction
        elif self.angle == -90:  # target is goal + turn_direction * 270
            goal = start_pose[1] - np.pi/2 * turn_direction
        elif self.angle == -100:  # target is goal + turn_direction * 270
            goal = start_pose[1] - 5*np.pi/9 * turn_direction
        elif self.angle == 120:
            goal = start_pose[1] + 2*np.pi/3 * turn_direction
        elif self.angle == 135:
            goal = start_pose[1] + 150*np.pi/180 * turn_direction

        goal = angles_lib.normalize_angle(goal)

        cur = np.abs(angles_lib.normalize_angle(self.tb_rot[2]) - goal)
        speed = 0.55
        rate = rospy.Rate(30)

        direction = turn_direction

        if 2 * np.pi - angles_lib.normalize_angle_positive(goal) < angles_lib.normalize_angle_positive(goal) or self.angle == 0:
            direction = turn_direction * -1

        while not rospy.is_shutdown():
            cur = np.abs(angles_lib.normalize_angle(self.tb_rot[2]) - goal)

            # slow down turning as we get closer to the target heading
            if cur < 0.1:
                speed = 0.15
            if cur < 0.0571:
                break
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = direction * speed
            self.cmd_pub.publish(msg)
            rate.sleep()

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

        return 'done'


class TurnAndFind(State):
    def __init__(self):
        State.__init__(self, outcomes=["find"],
                       output_keys=["current_marker"])
        self.rate = rospy.Rate(10)
        self.cmd_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.marker_sub = rospy.Subscriber(
            'ar_pose_marker_base', AlvarMarkers, self.marker_callback)
        self.marker_detected = False
        self.rate = rospy.Rate(30)

    def execute(self, userdata):
        global CURRENT_STATE, TAGS_FOUND
        CURRENT_STATE = "turn"

        self.marker_detected = False
        while not self.marker_detected:
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = -0.4
            self.cmd_pub.publish(msg)
            self.rate.sleep()

        userdata.current_marker = TAGS_FOUND[-1]
        self.marker_detected = False
        self.cmd_pub.publish(Twist())

        return "find"

    def marker_callback(self, msg):
        global TAG_POSE, TAGS_FOUND
        global CURRENT_STATE

        if CURRENT_STATE == "turn" and len(msg.markers) > 0:
            msg = msg.markers[0]

            if msg.id not in TAGS_FOUND:
                TAGS_FOUND.append(msg.id)
                TAG_POSE = msg.pose.pose
                self.marker_detected = True


class MoveBehind(State):
    def __init__(self):
        State.__init__(self, outcomes=["done"])
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)
        self.listener = tf.TransformListener()

    def execute(self, userdata):
        global TAG_POSE, TAGS_FOUND

        point_behind = PointStamped()
        point_behind.header.frame_id = "ar_marker_" + str(TAGS_FOUND[-1])
        point_behind.header.stamp = rospy.Time(0)
        point_behind.point.z = -1

        self.listener.waitForTransform(
            "odom", point_behind.header.frame_id, rospy.Time(0), rospy.Duration(4))
        point_behind_transformed = self.listener.transformPoint(
            "odom", point_behind)

        quaternion = quaternion_from_euler(0, 0, 0)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.pose.position.x = point_behind_transformed.point.x
        goal.target_pose.pose.position.y = point_behind_transformed.point.y
        goal.target_pose.pose.position.z = point_behind_transformed.point.z
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        self.move_base_client.send_goal_and_wait(goal)


class MoveBaseGo(State):
    def __init__(self, distance=0, horizontal=0, yaw=0, frame="base_footprint"):
        State.__init__(self, outcomes=["done"])
        self.distance = distance
        self.horizontal = horizontal
        self.yaw = yaw
        self.frame = frame
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)

    def execute(self, userdata):
        if START and not rospy.is_shutdown():

            quaternion = quaternion_from_euler(0, 0, self.yaw)

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = self.frame
            goal.target_pose.pose.position.x = self.distance
            goal.target_pose.pose.position.y = self.horizontal
            goal.target_pose.pose.orientation.x = quaternion[0]
            goal.target_pose.pose.orientation.y = quaternion[1]
            goal.target_pose.pose.orientation.z = quaternion[2]
            goal.target_pose.pose.orientation.w = quaternion[3]

            self.move_base_client.send_goal_and_wait(goal)

            return "done"


class MoveCloser(State):
    def __init__(self, ACAP=True):
        State.__init__(self, outcomes=["close_enough"], input_keys=[
                       "current_marker"])
        self.rate = rospy.Rate(10)
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.marker_sub = rospy.Subscriber(
            'ar_pose_marker_base', AlvarMarkers, self.marker_callback_base)

        self.current_marker = None
        self.tag_pose_base = None
        self.distance_from_marker = 0.2

        self.listener = tf.TransformListener()

    def execute(self, userdata):
        global CURRENT_POSE
        global CURRENT_STATE, START_POSE, END_GOAL, MARKER_POSE
        CURRENT_STATE = "move_closer"

        self.tag_pose_base = None
        self.current_marker = userdata.current_marker
        max_angular_speed = 0.8
        min_angular_speed = 0.0
        max_linear_speed = 0.8
        min_linear_speed = 0.0

        while True:
            if self.tag_pose_base is not None and self.tag_pose_base.position.x < 0.5:
                break
            elif self.tag_pose_base is not None and self.tag_pose_base.position.x > 0.5:
                move_cmd = Twist()

                if self.tag_pose_base.position.z > 0.6:  # goal too far
                    move_cmd.linear.x += 0.1
                elif self.tag_pose_base.position.z > 0.5:  # goal too close
                    move_cmd.linear.x -= 0.1
                else:
                    move_cmd.linear.x = 0

                if self.tag_pose_base.position.x < 1e-3:  # goal to the left
                    move_cmd.angular.z -= 0.1
                elif self.tag_pose_base.position.x > -1e-3:  # goal to the right
                    move_cmd.angular.z += 0.1
                else:
                    move_cmd.angular.z = 0

                move_cmd.linear.x = math.copysign(max(min_linear_speed, min(
                    abs(move_cmd.linear.x), max_linear_speed)), move_cmd.linear.x)
                move_cmd.angular.z = math.copysign(max(min_angular_speed, min(
                    abs(move_cmd.angular.z), max_angular_speed)), move_cmd.angular.z)

                move_cmd.linear.x = abs(move_cmd.linear.x)

                self.vel_pub.publish(move_cmd)
            self.rate.sleep()

        pose = PointStamped()
        pose.header.frame_id = "ar_marker_" + str(self.current_marker)
        pose.header.stamp = rospy.Time(0)
        pose.point.z = self.distance_from_marker

        # TODO: Change to -0.1?????
        pose.point.x = 0.1

        self.listener.waitForTransform(
            "odom", pose.header.frame_id, rospy.Time(0), rospy.Duration(4))

        pose_transformed = self.listener.transformPoint("odom", pose)

        if ACAP:
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "odom"
            goal.target_pose.pose.position.x = pose_transformed.point.x
            goal.target_pose.pose.position.y = pose_transformed.point.y
            goal.target_pose.pose.orientation = START_POSE.orientation
            END_GOAL = goal
        else:
            MARKER_POSE = self.tag_pose_base
        return "close_enough"

    def marker_callback_base(self, msg):
        global CURRENT_STATE
        if CURRENT_STATE == "move_closer" and self.current_marker is not None:
            for marker in msg.markers:
                if marker.id == self.current_marker:
                    self.tag_pose_base = marker.pose.pose


def odom_callback(msg):
    global CURRENT_POSE, START_POSE

    CURRENT_POSE = msg.pose.pose
    if START_POSE == None:
        START_POSE = CURRENT_POSE


if __name__ == "__main__":
    rospy.init_node('demo6')

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    sm = StateMachine(outcomes=['success', 'failure'])
    sm.userdata.goal = None

    rospy.Subscriber("odom", Odometry, callback=odom_callback)

    with sm:

        StateMachine.add("GoToMiddle", MoveBaseGo(
            1.5, -1, 0, "base_link"), transitions={"done": "Turn"})

        StateMachine.add("Turn", Turn(90), transitions={"done": "FindAR"})

        StateMachine.add("FindAR", TurnAndFind(),
                         transitions={"find": "GetCloseToAR"})

        StateMachine.add("GetCloseToAR", MoveCloser(), transitions={
                         "close_enough": "GoToBackwardMiddle"})

        StateMachine.add("GoToBackwardMiddle", MoveBaseGo(-1, 0, math.pi/2, "base_link"),
                         transitions={"done": "FindBox"})

        StateMachine.add("FindBox", TurnAndFind(), transitions={
                         "find": "MoveBehind"})

        # StateMachine.add("GetCloseToBox", MoveCloser(False),
        #                  transitions={"close_enough": "GoToSide"})

        # StateMachine.add("GoToSide", , transitions={"done": "MoveBehind"})

        StateMachine.add("MoveBehind", MoveBehind(),
                         transitions={"done": "success"})

        # StateMachine.add("TouchBox", , transitions={"done": "GoToGoal"})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()
    outcome = sm.execute()
    sis.stop()
