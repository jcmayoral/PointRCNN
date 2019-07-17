import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import numpy as np
from sensor_msgs import point_cloud2
from std_msgs.msg import Header

class BoundingBoxPublisher(rospy.Publisher):
    def __init__(self, topic_name, msg_class = BoundingBoxArray):
        rospy.Publisher.__init__(self, topic_name, msg_class, queue_size=1)
        self.array = BoundingBoxArray()

    def euler_to_quaternion(self,yaw, pitch, roll):
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

    def add_bb(self, data, stamp=None, frame_id=None):
        """
        :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
        :return:
            boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
        """
        msg = BoundingBox()

        if stamp:
            msg.header.stamp = stamp
        if frame_id:
            msg.header.frame_id = frame_id

        msg.pose.position.x = data[0].item()
        msg.pose.position.y = data[1].item()
        msg.pose.position.z = data[2].item()

        quaternion = self.euler_to_quaternion(0, 0, data[6].item())
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]

        msg.dimensions.x = data[3].item()
        msg.dimensions.y = data[4].item()
        msg.dimensions.z = data[5].item()

        self.array.boxes.append(msg)

    def custom_publish(self):
        self.array.header.stamp = rospy.Time.now()
        self.array.header.frame_id = "velodyne"

        self.publish(self.array)
        rospy.loginfo("publishing bb")
