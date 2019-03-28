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

isToTheLeft = False

POINT_BEHIND_TRANS = None
PARK_MARK = 0


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
        global isToTheLeft
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

        elif self.angle == 999:
            if isToTheLeft:
                goal = start_pose[1] - np.pi/2 * turn_direction
            else:
                goal = start_pose[1] + np.pi/2 * turn_direction

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
        self.count = 0

    def execute(self, userdata):
        global CURRENT_STATE, TAGS_FOUND
        CURRENT_STATE = "turn"
        self.count = 0

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
            self.count += 1

            # if msg.id not in TAGS_FOUND:
            if self.count > 2:
                TAGS_FOUND.append(msg.id)
                TAG_POSE = msg.pose.pose
                self.marker_detected = True


def calc_delta_vector(start_heading, distance):
    dx = distance * np.cos(start_heading)
    dy = distance * np.sin(start_heading)
    return np.array([dx, dy])


def check_forward_distance(forward_vec, start_pos, current_pos):
    current = current_pos - start_pos
    # vector projection (project current onto forward_vec)
    delta = np.dot(current, forward_vec) / \
        np.dot(forward_vec, forward_vec) * forward_vec
    dist = np.sqrt(delta.dot(delta))
    return dist


class Translate(State):
    def __init__(self, distance=0.15, linear=-0.2, mode=0):
        State.__init__(self, outcomes=["done"])
        self.tb_position = None
        self.tb_rot = [0, 0, 0, 0]
        self.distance = distance
        self.COLLISION = False
        self.linear = linear
        self.mode = mode

        self.listener = tf.TransformListener()

        # pub / sub
        self.cmd_pub = rospy.Publisher(
            "cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("odom", Odometry, callback=self.odom_callback)

    def odom_callback(self, msg):
        tb_pose = msg.pose.pose
        __, __, angles, position, __ = decompose_matrix(numpify(tb_pose))
        self.tb_position = position[0:2]
        self.tb_rot = angles

    def execute(self, userdata):
        global turn_direction
        global START
        global CURRENT_POSE
        global END_GOAL, POINT_BEHIND_TRANS, isToTheLeft

        if not START:
            return 'quit'
        self.COLLISION = False
        start_heading = self.tb_rot[2]
        start_pos = self.tb_position

        if self.mode == 1:
            self.distance = abs(
                CURRENT_POSE.position.y - END_GOAL.target_pose.pose.position.y) - 0.15
            self.linear = 0.2
        elif self.mode == 2:
            self.distance = abs(
                CURRENT_POSE.position.x - END_GOAL.target_pose.pose.position.x) - 0.2
            self.linear = 0.2

        forward_vec = calc_delta_vector(start_heading, self.distance)
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            dist = check_forward_distance(
                forward_vec, start_pos, self.tb_position)
            if dist > self.distance:
                if self.mode == 0:
                    if isToTheLeft:
                        delta_x = 1
                    else:
                        delta_x = -1

                    # calculate point behind
                    point_behind = PointStamped()
                    point_behind.header.frame_id = "ar_marker_" + \
                        str(TAGS_FOUND[-1])
                    point_behind.header.stamp = rospy.Time(0)
                    point_behind.point.z = -0.2
                    point_behind.point.x = delta_x

                    self.listener.waitForTransform(
                        "odom", point_behind.header.frame_id, rospy.Time(0), rospy.Duration(4))

                    POINT_BEHIND_TRANS = self.listener.transformPoint(
                        "odom", point_behind)

                return "done"

            msg = Twist()
            msg.linear.x = self.linear
            self.cmd_pub.publish(msg)
            rate.sleep()


class MoveBehind(State):
    def __init__(self):
        State.__init__(self, outcomes=["done"])
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)
        self.listener = tf.TransformListener()

    def execute(self, userdata):
        global TAG_POSE, TAGS_FOUND, isToTheLeft, POINT_BEHIND_TRANS

        # point_behind = PointStamped()
        # point_behind.header.frame_id = "ar_marker_" + str(TAGS_FOUND[-1])
        # point_behind.header.stamp = rospy.Time(0)
        # point_behind.point.z = -0.2
        # point_behind.point.x = delta_x

        # self.listener.waitForTransform(
        #     "odom", point_behind.header.frame_id, rospy.Time(0), rospy.Duration(4))
        # point_behind_transformed = self.listener.transformPoint(
        #     "odom", point_behind)

        quaternion = quaternion_from_euler(0, 0, 0)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.pose.position.x = POINT_BEHIND_TRANS.point.x
        goal.target_pose.pose.position.y = POINT_BEHIND_TRANS.point.y
        goal.target_pose.pose.position.z = POINT_BEHIND_TRANS.point.z
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        self.move_base_client.send_goal_and_wait(goal)

        return "done"


class MoveBaseGo(State):
    def __init__(self, distance=0, horizontal=0, yaw=0, frame="base_link", isNotPushing=0):
        State.__init__(self, outcomes=["done"])
        self.distance = distance
        self.horizontal = horizontal
        self.yaw = yaw
        self.frame = frame
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)

        self.isNotPushing = isNotPushing

    def execute(self, userdata):
        global CURRENT_POSE, START, END_GOAL, isToTheLeft

        if START and not rospy.is_shutdown():

            if self.isNotPushing == 0:
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
            elif self.isNotPushing == 1:
                rospy.set_param(
                    "/move_base/DWAPlannerROS/yaw_goal_tolerance", 2*math.pi)
                rospy.set_param(
                    "/move_base/DWAPlannerROS/latch_xy_goal_tolerance", True)

                if isToTheLeft:
                    angle = -math.pi/2
                else:
                    angle = math.pi/2
                quaternion = quaternion_from_euler(0, 0, angle)

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "base_link"
                goal.target_pose.pose.position.x = abs(
                    CURRENT_POSE.position.y - END_GOAL.target_pose.pose.position.y) - 0.15  # TODO: adjust this value
                goal.target_pose.pose.position.y = 0
                goal.target_pose.pose.orientation.x = quaternion[0]
                goal.target_pose.pose.orientation.y = quaternion[1]
                goal.target_pose.pose.orientation.z = quaternion[2]
                goal.target_pose.pose.orientation.w = quaternion[3]

                self.move_base_client.send_goal_and_wait(goal)
                rospy.set_param(
                    "/move_base/DWAPlannerROS/yaw_goal_tolerance", 0.3)
                rospy.set_param(
                    "/move_base/DWAPlannerROS/latch_xy_goal_tolerance", False)
            elif self.isNotPushing == 2:
                rospy.set_param(
                    "/move_base/DWAPlannerROS/yaw_goal_tolerance", 2*math.pi)
                rospy.set_param(
                    "/move_base/DWAPlannerROS/latch_xy_goal_tolerance", True)

                quaternion = quaternion_from_euler(0, 0, 0)

                goal = MoveBaseGoal()
                goal.target_pose.header.frame_id = "base_link"
                goal.target_pose.pose.position.x = abs(
                    CURRENT_POSE.position.x - END_GOAL.target_pose.pose.position.x) - 0.3  # TODO: adjust this value
                goal.target_pose.pose.position.y = 0
                goal.target_pose.pose.orientation.x = quaternion[0]
                goal.target_pose.pose.orientation.y = quaternion[1]
                goal.target_pose.pose.orientation.z = quaternion[2]
                goal.target_pose.pose.orientation.w = quaternion[3]

                self.move_base_client.send_goal_and_wait(goal)
                rospy.set_param(
                    "/move_base/DWAPlannerROS/yaw_goal_tolerance", 0.3)
                rospy.set_param(
                    "/move_base/DWAPlannerROS/latch_xy_goal_tolerance", False)
            return "done"


class StopInFront(State):
    def __init__(self, distance=-0.2):
        State.__init__(self,  outcomes=["done"])
        self.marker_sub = rospy.Subscriber(
            'ar_pose_marker_base', AlvarMarkers, self.marker_callback_base)
        self.tag_pose_base = None
        self.distance = distance
        self.listener = tf.TransformListener()
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)

    def execute(self, userdata):
        global POINT_BEHIND_TRANS, isToTheLeft

        self.tag_pose_base = None
        while self.tag_pose_base == None:
            continue

        point = PointStamped()
        point.header.frame_id = "ar_marker_" + str(TAGS_FOUND[-1])
        point.header.stamp = rospy.Time(0)
        point.point.z = self.distance  # TODO: maybe change it

        self.listener.waitForTransform(
            "odom", point.header.frame_id, rospy.Time(0), rospy.Duration(4))

        point_transformed = self.listener.transformPoint(
            "odom", point)

        if isToTheLeft:
            angle = -math.pi/2
        else:
            angle = math.pi/2

        quaternion = quaternion_from_euler(0, 0, angle)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.pose.position.x = point_transformed.point.x
        goal.target_pose.pose.position.y = point_transformed.point.y
        goal.target_pose.pose.position.z = point_transformed.point.z
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        self.move_base_client.send_goal_and_wait(goal)

        return "done"

    def marker_callback_base(self, msg):
        if msg.markers:
            self.tag_pose_base = msg.markers[0].pose.pose


class MoveCloser(State):
    def __init__(self, ACAP=True, how_close=0.5):
        State.__init__(self, outcomes=["close_enough"], input_keys=[
                       "current_marker"])
        self.rate = rospy.Rate(10)
        self.vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.marker_sub = rospy.Subscriber(
            'ar_pose_marker_base', AlvarMarkers, self.marker_callback_base)

        self.current_marker = None
        self.tag_pose_base = None
        self.distance_from_marker = 0.2
        self.ACAP = ACAP
        self.how_close = how_close
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
            if self.tag_pose_base is not None and self.tag_pose_base.position.x < self.how_close:
                break
            elif self.tag_pose_base is not None and self.tag_pose_base.position.x > self.how_close:
                move_cmd = Twist()

                if self.tag_pose_base.position.x > self.how_close+0.1:  # goal too far
                    move_cmd.linear.x += 0.1
                elif self.tag_pose_base.position.x > self.how_close:  # goal too close
                    move_cmd.linear.x -= 0.1
                else:
                    move_cmd.linear.x = 0

                if self.tag_pose_base.position.y < 1e-3:  # goal to the left
                    move_cmd.angular.z -= 0.1
                elif self.tag_pose_base.position.y > -1e-3:  # goal to the right
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

        print("marker!!!:", self.current_marker)
        pose = PointStamped()
        pose.header.frame_id = "ar_marker_" + str(self.current_marker)
        pose.header.stamp = rospy.Time(0)
        pose.point.z = self.distance_from_marker

        # TODO: Change to -0.1?????
        pose.point.x = 0.1

        self.listener.waitForTransform(
            "odom", pose.header.frame_id, rospy.Time(0), rospy.Duration(4))

        pose_transformed = self.listener.transformPoint("odom", pose)

        if self.ACAP:
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "odom"
            goal.target_pose.pose.position.x = pose_transformed.point.x
            goal.target_pose.pose.position.y = pose_transformed.point.y
            goal.target_pose.pose.orientation = START_POSE.orientation
            END_GOAL = goal
            PARK_MARK = self.current_marker
        else:
            MARKER_POSE = self.tag_pose_base
        return "close_enough"

    def marker_callback_base(self, msg):
        global CURRENT_STATE
        if CURRENT_STATE == "move_closer" and self.current_marker is not None:
            for marker in msg.markers:
                if marker.id == self.current_marker:
                    self.tag_pose_base = marker.pose.pose


class MoveToSide(State):
    def __init__(self):
        State.__init__(self, outcomes=["done"])
        self.listener = tf.TransformListener()
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction)

    def execute(self, userdata):
        global END_GOAL, TAG_POSE, TAGS_FOUND, isToTheLeft

        # compare if the tag is to the left of the end go pose

        # convert TAG to odom
        tag_point = PointStamped()
        tag_point.header.frame_id = "ar_marker_" + str(TAGS_FOUND[-1])
        tag_point.header.stamp = rospy.Time(0)

        self.listener.waitForTransform(
            "odom", tag_point.header.frame_id, rospy.Time(0), rospy.Duration(4))

        tag_point_transformed = self.listener.transformPoint("odom", tag_point)

        isToTheLeft = tag_point_transformed.point.y > END_GOAL.target_pose.pose.position.y

        if isToTheLeft:
            delta_x = 1.5
            angle = -math.pi/2
        else:
            delta_x = -1.5
            angle = math.pi/2

        side_point = PointStamped()
        side_point.header.frame_id = "ar_marker_" + str(TAGS_FOUND[-1])
        side_point.header.stamp = rospy.Time(0)
        side_point.point.x = delta_x
        side_point.point.z = -0.15

        side_point_transformed = self.listener.transformPoint(
            "odom", side_point)

        quaternion = quaternion_from_euler(0, 0, angle)
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.pose.position.x = side_point_transformed.point.x
        goal.target_pose.pose.position.y = side_point_transformed.point.y
        goal.target_pose.pose.position.z = side_point_transformed.point.z
        goal.target_pose.pose.orientation.x = quaternion[0]
        goal.target_pose.pose.orientation.y = quaternion[1]
        goal.target_pose.pose.orientation.z = quaternion[2]
        goal.target_pose.pose.orientation.w = quaternion[3]

        self.move_base_client.send_goal_and_wait(goal)

        return "done"


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
            1.5, -0.8, 0, "base_link"), transitions={"done": "Turn"})

        StateMachine.add("Turn", Turn(90), transitions={"done": "FindAR"})

        StateMachine.add("FindAR", TurnAndFind(),
                         transitions={"find": "GetCloseToAR"})

        StateMachine.add("GetCloseToAR", MoveCloser(), transitions={
                         "close_enough": "GoToBackwardMiddle"})

        # StateMachine.add("SeanTurnSaveMe", Turn(0), transitions={
        #                  "done": "GoToBackwardMiddle"})

        StateMachine.add("GoToBackwardMiddle", MoveBaseGo(-1, 0, -math.pi/2, "base_link"),
                         transitions={"done": "FindBox"})

        StateMachine.add("FindBox", TurnAndFind(), transitions={
                         "find": "GetClose"})

        StateMachine.add("GetClose", MoveCloser(False, 1),
                         transitions={"close_enough": "MoveToSide"})

        StateMachine.add("MoveToSide", MoveToSide(),
                         transitions={"done": "StopInFront"})

        StateMachine.add("StopInFront", StopInFront(0.2),
                         transitions={"done": "SeanTurnSide"})

        StateMachine.add("SeanTurnSide", Turn(
            999), transitions={"done": "Straight"})

        StateMachine.add("Straight", Translate(0, 0, 1),
                         transitions={"done": "MoveBack"})

        StateMachine.add("MoveBack", Translate(1),
                         transitions={"done": "SeanTurn180"})

        StateMachine.add("SeanTurn180", Turn(
            180), transitions={"done": "MoveBehind"}),

        StateMachine.add("MoveBehind", MoveBehind(),
                         transitions={"done": "SeanTurn0_1"})

        StateMachine.add("SeanTurn0_1", Turn(
            0), transitions={"done": "StopInFront2"})

        StateMachine.add("StopInFront2", StopInFront(0.2),
                         transitions={"done": "SeanTurn0"})

        StateMachine.add("SeanTurn0", Turn(
            0), transitions={"done": "Straight2"})

        StateMachine.add("Straight2", MoveBaseGo(
            0, 0, 0, "base_link", 2), transitions={"done": "success"})

        # StateMachine.add("TouchBox", , transitions={"done": "GoToGoal"})

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()
    outcome = sm.execute()
    sis.stop()
