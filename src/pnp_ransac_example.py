#!/usr/bin/env python

import cv2
import yaml
import numpy as np
import glob
import os
import rospy
import cv_bridge
from sensor_msgs.msg import Joy, LaserScan, Image
import math
import random
import imutils

criteria = None
axis = None

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def image_callback(msg):
    print "hello"
    global bridge, criteria, axis

    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

    if ret == True and criteria is not None and axis is not None:
        # refine corners
        corners2 = cv2.cornerSubPix(gray, corners,(8,6),(-1,-1), criteria)

        # Find the rotation and translation vectors.
        success, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners2,imgpts)

    cv2.imshow('img', img)
    cv2.waitKey(3)

if __name__ == "__main__":
    rospy.init_node('demo5p1')
    bridge = cv_bridge.CvBridge()
    image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, image_callback)

    mtx = None
    dist = None

    with open(os.path.join(os.environ['HOME'],".ros/camera_info","rgb_PS1080_PrimeSense.yaml"), 'r') as stream:
        try:
            out = yaml.load(stream)
            mtx = np.matrix(out['camera_matrix']['data']).reshape((3,3))
            dist = np.array(out['distortion_coefficients']['data'])
            print(mtx)
        except yaml.YAMLError as exc:
            print(exc)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    print(objp)

    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

    rospy.spin()


