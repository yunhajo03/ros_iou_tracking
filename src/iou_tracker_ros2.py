#!/usr/bin/env python3
"""
    ros_iou_tracking
    Copyright 2021 Zhiang Chen


    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at


       http://www.apache.org/licenses/LICENSE-2.0


    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""


import rclpy
from rclpy.node import Node
# from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from vision_msgs.msg import BoundingBox2D, Detection2DArray, Detection2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import copy
import numpy as np
import cv2
from sort import Sort
from sort import associate_detections_to_trackers


class IoUTracker(Node):
    def __init__(self, max_age=100, min_hits=10, iou_threshold=0.3):
        super().__init__('iou_tracker')
        """
        ROS IoU Tracker
        :param max_age: Maximum number of frames to keep alive a track without associated detections.
        :param min_hits: Minimum number of associated detections before track is initialised.
        :param iou_threshold: Minimum IOU for match.
        """
        self.iou_threshold = iou_threshold
        self.bridge = CvBridge()
        self.tracked_img_pub = self.create_publisher(Image, "/iou_tracker/detection_image", 10)
        self.new_bboxes = []
        self.bboxes = []
        self.bboxes_msg = BoundingBox2D()
        self.traces = dict()
        self.mot_tracker = Sort(max_age=max_age,
                                min_hits=min_hits,
                                iou_threshold=iou_threshold)  # create instance of the SORT tracker
        self.image = np.zeros(1)
        self.raw_image_sub = self.create_subscription(Image, '/camera', self.__raw_image_callback, 10)
        #self.raw_image_sub = self.create_subscription(Image, '/r200/depth/image_raw', self.__raw_image_callback, 10)


        self.bbox_pub = self.create_publisher(Detection2DArray, "/iou_tracker/bounding_boxes", 10)
        self.bbox_nn_sub = self.create_subscription(Detection2DArray, '/yolov7_detections', self.__bbox_nn_callback, 10)
        self.get_logger().info("iou_tracker has been initialized!")


    def __bbox_nn_callback(self, data):
        """
        ROS2 message structure:
        std_msgs/Header header
        vision_msgs/Detection2D[] detections
            std_msgs/Header header
            vision_msgs/ObjectHypothesisWithPose[] results
                int64 id
                float64 score
                geometry_msgs/PoseWithCovariance pose
            vision_msgs/BoundingBox2D bbox
                geometry_msgs/Pose2D center
                float64 size_x
                float64 size_y
            sensor_msgs/Image source_img
        :return:
        """
        self.new_bboxes = data.detections  # new_bboxes is a list of darknet_ros_msgs.msg.BoundingBox


    def __publish_tracking_image(self, image, bboxes):
        Xs = copy.deepcopy(bboxes)
        if len(image.shape) <= 2:
            return
        for x in Xs:
            xmin = x.bbox.center.x - x.bbox.size_x / 2
            xmax = x.bbox.center.x + x.bbox.size_x / 2
            ymin = x.bbox.center.y - x.bbox.size_y / 2
            ymax = x.bbox.center.y + x.bbox.size_y / 2
            image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)


        for k in self.traces:
            trace = self.traces[k]
            if len(trace) >= 200:
                trace = trace[-200:]
                self.traces[k] = trace
            pts = copy.deepcopy(trace)
            pts.reverse()
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                buffer = 32
                thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
                cv2.line(image, pts[i - 1], pts[i], (255, 255, 0), thickness)


        image_msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        self.tracked_img_pub.publish(image_msg)


    def __publish_bbox(self):
        if len(self.bboxes) != 0:
            if len(self.image.shape) > 2:
                self.bboxes_msg.header = self.image_header
                # self.bboxes_msg.image_header = self.image_header
                self.bboxes_msg.detections = copy.deepcopy(self.bboxes)
                self.bbox_pub.publish(self.bboxes_msg)


    def __raw_image_callback(self, data):
        self.image_header = data.header
        raw_image = self.bridge.imgmsg_to_cv2(data).astype(np.uint8)
        if len(raw_image.shape) <= 2:
            return
        self.image = raw_image


        # 1. get SORT bounding boxes


        new_bboxes_sort = []
        if len(self.new_bboxes) != 0:
            new_bboxes = copy.deepcopy(self.new_bboxes)
            self.new_bboxes = []
            for bbox in new_bboxes:
                xmin = bbox.bbox.center.x - bbox.bbox.size_x / 2
                xmax = bbox.bbox.center.x + bbox.bbox.size_x / 2
                ymin = bbox.bbox.center.y - bbox.bbox.size_y / 2
                ymax = bbox.bbox.center.y + bbox.bbox.size_y / 2
                probability = bbox.results.score
                new_bboxes_sort.append(np.asarray((xmin, ymin, xmax, ymax, probability)))
            new_bboxes_sort = np.asarray(new_bboxes_sort)
            # 2. update tracker
            trackers = self.mot_tracker.update(new_bboxes_sort)
            matched, _, _ = associate_detections_to_trackers(new_bboxes_sort, trackers, 0.3)


        else:
            # 2. update tracker
            trackers = self.mot_tracker.update()
            if len(self.bboxes) == 0:
                matched = np.empty((0, 2), dtype=int)
            else:
                new_bboxes = copy.deepcopy(self.bboxes)
                for bbox in new_bboxes:
                    xmin = bbox.bbox.center.x - bbox.bbox.size_x / 2
                    xmax = bbox.bbox.center.x + bbox.bbox.size_x / 2
                    ymin = bbox.bbox.center.y - bbox.bbox.size_y / 2
                    ymax = bbox.bbox.center.y + bbox.bbox.size_y / 2
                    probability = bbox.results.score
                    new_bboxes_sort.append(np.asarray((xmin, ymin, xmax, ymax, probability)))
                new_bboxes_sort = np.asarray(new_bboxes_sort)
                matched, _, _ = associate_detections_to_trackers(new_bboxes_sort, trackers, 0.3)


        # 3. update current bounding boxes & extract tracking trace
        # 1). id is the id from tracker;
        # 2). class is the latest class from detection;
        # 3). probability is the latest probability from detection;
        self.bboxes = []
        ids = []
        if trackers.shape[0] != 0:
            for tracker in trackers:
                print(tracker)
                xmin, ymin, xmax, ymax = int(tracker[0]), int(tracker[1]), int(tracker[2]), int(tracker[3])
                id = int(tracker[4])
                center = (int((xmin + xmax) / 2.), int((ymin + ymax) / 2.))
                if self.traces.get(id) is None:
                    self.traces[id] = [center]
                else:
                    self.traces[id].append(center)
                ids.append(id)

                detect_msg = Detection2D()

                detect_msg.header = header()

                detect_msg.bbox = BoundingBox2D()
                detect_msg.bbox.center = Pose2D()
                detect_msg.bbox.center.x = center[0]
                detect_msg.bbox.center.y = center[1]
                detect_msg.bbox.size_x = xmax - xmin
                detect_msg.bbox.size_y = ymax - ymin

                obj_hyp = ObjectHypothesisWithPose()
                obj_hyp.hypothesis = ObjectHypothesis()
                obj_hyp.hypothesis.class_id = str(id)
                detect_msg.results = [obj_hyp]

                self.bboxes.detections.append(detect_msg)


        if matched.shape[0] != 0:
            for pair in matched:
                original_id = pair[0]
                new_id = pair[1]
                original_bbox = new_bboxes[original_id]
                # self.bboxes[new_id].Class = original_bbox.Class
                self.bboxes[new_id].results.score = original_bbox.results.score


        del_ids = []
        for k in self.traces:
            if k not in ids:
                del_ids.append(k)


        for k in del_ids:
            del self.traces[k]


        # 4. publish current bounding boxes
        self.__publish_bbox()


        # 5. publish tracking image
        self.__publish_tracking_image(self.image, self.bboxes)




def main(args=None):
    rclpy.init(args=args)
    iou_tracker = IoUTracker()
    try:
        rclpy.spin(iou_tracker)
    except rclpy.exceptions.ROSInterruptException:
        iou_tracker.get_logger().info("Node killed!")
    finally:
        iou_tracker.destroy_node()
        rclpy.shutdown()




if __name__ == '__main__':
    main()
