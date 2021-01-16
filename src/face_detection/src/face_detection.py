#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import datetime
import os
import time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import rospkg 


rospack = rospkg.RosPack()
path = rospack.get_path('face_detection')

bridge = CvBridge()
protoPath = path + "/model/deploy.prototxt.txt"
modelPath = path + "/model/res10_300x300_ssd_iter_140000.caffemodel"
detectors = cv2.dnn.readNet(protoPath,modelPath)

def main():

    rospy.init_node("usb_camera_publisher",anonymous=True)
    img_pub = rospy.Publisher("/camera/image_raw",Image,queue_size=10)
    face_pub = rospy.Publisher("/camera/detect/image",Image,queue_size=10)

    rate = rospy.Rate(30)

    video = cv2.VideoCapture(0)

    while not rospy.is_shutdown():

        ret, img = video.read()

        # Publish original image
        img_raw = bridge.cv2_to_imgmsg(img,"bgr8")
        img_raw.header.stamp = rospy.Time.now()
        img_pub.publish(img_raw)

        # Face Detection
        (h,w) = img.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        detectors.setInput(imageBlob)
        detections = detectors.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
            if confidence > 0.8:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
        
                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(img, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                cv2.putText(img, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # To show the output image
        # cv2.imshow("Output", img)
        # cv2.waitKey(1)

        face_img = bridge.cv2_to_imgmsg(img,"bgr8")
        face_img.header.stamp = rospy.Time.now()
        face_pub.publish(face_img)

        rate.sleep()

    print("release capture")
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


