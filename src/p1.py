#!/usr/bin/env python

import cv2
import yaml
import numpy as np
import glob
import os
import rospy
import cv_bridge
from geometry_msgs.msg import Twist, Pose, PoseStamped, PointStamped
from sensor_msgs.msg import Joy, LaserScan, Image, CameraInfo
from ar_track_alvar_msgs.msg import AlvarMarkers
import math
import random
import imutils
import tf
import image_geometry

def image_callback(msg):
    global bridge, listener, current_marker

    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    if current_marker["id"] is not None:
        listener.waitForTransform("ar_marker_" + str(current_marker["id"]), cam_model.tfFrame(), rospy.Time(0), rospy.Duration(4))
        # position, quaternion = listener.lookupTransform(cam_model.tfFrame(), "ar_marker_" + str(current_marker["id"]), rospy.Time(0))

        origin = PointStamped()
        origin.header.frame_id = "ar_marker_" + str(current_marker["id"])
        origin.header.stamp = rospy.Time(0)
        origin.point.x = 0.0
        origin.point.y = 0.0
        origin.point.z = 0.0

        x_axis = PointStamped()
        x_axis.header.frame_id = "ar_marker_" + str(current_marker["id"])
        x_axis.header.stamp = rospy.Time(0)
        x_axis.point.x = 0.05
        x_axis.point.y = 0.0
        x_axis.point.z = 0.0

        y_axis = PointStamped()
        y_axis.header.frame_id = "ar_marker_" + str(current_marker["id"])
        y_axis.header.stamp = rospy.Time(0)
        y_axis.point.x = 0.0
        y_axis.point.y = 0.05
        y_axis.point.z = 0.0

        z_axis = PointStamped()
        z_axis.header.frame_id = "ar_marker_" + str(current_marker["id"])
        z_axis.header.stamp = rospy.Time(0)
        z_axis.point.x = 0.0
        z_axis.point.y = 0.0
        z_axis.point.z = 0.05

        origin_transformed = listener.transformPoint(cam_model.tfFrame(), origin)
        x_axis_transformed = listener.transformPoint(cam_model.tfFrame(), x_axis)
        y_axis_transformed = listener.transformPoint(cam_model.tfFrame(), y_axis)
        z_axis_transformed = listener.transformPoint(cam_model.tfFrame(), z_axis)

        uv_origin = cam_model.project3dToPixel((origin_transformed.point.x, origin_transformed.point.y, origin_transformed.point.z))
        uv_x = cam_model.project3dToPixel((x_axis_transformed.point.x, x_axis_transformed.point.y, x_axis_transformed.point.z))
        uv_y = cam_model.project3dToPixel((y_axis_transformed.point.x, y_axis_transformed.point.y, y_axis_transformed.point.z))
        uv_z = cam_model.project3dToPixel((z_axis_transformed.point.x, z_axis_transformed.point.y, z_axis_transformed.point.z))

        cv2.line(img, (int(uv_origin[0]), int(uv_origin[1])), (int(uv_x[0]), int(uv_x[1])), (255,0,0), 5)
        cv2.line(img, (int(uv_origin[0]), int(uv_origin[1])), (int(uv_y[0]), int(uv_y[1])), (0,255,0), 5)
        cv2.line(img, (int(uv_origin[0]), int(uv_origin[1])), (int(uv_z[0]), int(uv_z[1])), (0,0,255), 5)


    cv2.imshow('img', img)
    cv2.waitKey(3)

def marker_callback(msg):
    global current_marker

    if len(msg.markers) > 0:
        msg = msg.markers[0]

        current_marker["pose"] = msg.pose.pose
        current_marker["id"] = msg.id
    else:
        current_marker["pose"] = None
        current_marker["id"] = None

if __name__ == "__main__":
    rospy.init_node("demo6p1")

    current_marker = {"pose": None, "id": None}

    bridge = cv_bridge.CvBridge()
    cam_model = image_geometry.PinholeCameraModel()
    listener = tf.TransformListener()
    
    info_msg = rospy.wait_for_message("/camera/rgb/camera_info", CameraInfo)
    cam_model.fromCameraInfo(info_msg)
    print "got camera info"

    # listener.waitForTransform("odom", cam_model.tfFrame(), rospy.Time(0), rospy.Duration(4))

    image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, image_callback)
    marker_sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, marker_callback)

    rospy.spin()


