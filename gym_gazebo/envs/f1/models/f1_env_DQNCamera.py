import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from gym import spaces
from gym.utils import seeding
from sensor_msgs.msg import Image

#from agents.f1.settings import telemetry, x_row, center_image, width, height, telemetry_mask, max_distance
from settings import x_row, center_image, width, height, telemetry_mask, max_distance
from gym_gazebo.envs.f1.image_f1 import ImageF1
from gym_gazebo.envs.f1.models.f1_env import F1Env

from cprint import cprint
from icecream import ic
from datetime import datetime
import time


ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')

class F1DQNCameraEnv(F1Env):

    def __init__(self, **config):

        #cprint.warn(f"\n [F1DQNCameraEnv] -> --------- Enter in F1DQNCameraEnv ---------------\n")
        #ic('Enter in F1DQNCameraEnv')
        F1Env.__init__(self, **config)
        #print(f"\n [F1DQNCameraEnv] -> config: {config}")
        self.image = ImageF1()
        self.actions = config.get("actions")
        self.action_space = spaces.Discrete(len(self.actions))  # actions  # spaces.Discrete(3)  # F,L,R

        self.rewards = config["rewards"]
        self.telemetry = config["telemetry"]
        #ic(self.rewards)
        #ic(self.rewards['from_done'])


        #cprint.ok(f"\n  [F1DQNCameraEnv] -> ------------ Out F1DQNCameraEnv (__init__) -----------\n")


    def reset(self):
        #print(f"\n F1QlearnCameraEnv.reset()\n")
        #ic("F1DQNCameraEnv.reset()")

        #F1Env.__init__(self, **config)
        #ic("salimos")

        # === POSE ===
        if self.alternate_pose:
            #print(f"alternate_pose:{self.alternate_pose}")
            #self._gazebo_set_new_pose()
            #self._gazebo_reset_random_pose()
            self._gazebo_set_new_pose()
            #self.set_new_pose()
        else:
            self._gazebo_reset()

        self._gazebo_unpause()

        # Get camera info
        #ic("volvemos a F1DQNCameraEnv.reset()")
        image_data = None
        f1_image_camera = None
        success = False
        while image_data is None or success is False:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=None)
            #time.sleep(3)
            #image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
            #f1_image_camera = cv_image
            if np.any(cv_image):
                success = True

        #points = self.processed_image(f1_image_camera.data)
        #state = self.calculate_observation(points)
        # reset_state = (state, False)


        # veamos que center me da en el reset()
        points = self.processed_image(f1_image_camera.data)
        center = float(center_image - points[0]) / (float(width) // 2)
        #done = False
        center = abs(center)
        if self.telemetry:
            print(f"\n F1DQNCameraEnv.reset() -> center: {center}")

        self._gazebo_pause()

        #return state
        return np.array(cv_image)


    def image_msg_to_image(self, img, cv_image):
        #ic("F1DQNCameraEnv.image_msg_to_image()")
        #print(f"\n F1QlearnCameraEnv.image_msg_to_image()\n")

        self.image.width = img.width
        self.image.height = img.height
        self.image.format = "RGB8"
        self.image.timeStamp = img.header.stamp.secs + (img.header.stamp.nsecs * 1e-9)
        self.image.data = cv_image

        return self.image        


    @staticmethod
    def get_center(lines):
        #ic("F1DQNCameraEnv.get_center()")
        #print(f"\n F1QlearnCameraEnv.get_center()\n")
        try:
            point = np.divide(np.max(np.nonzero(lines)) - np.min(np.nonzero(lines)), 2)
            point = np.min(np.nonzero(lines)) + point
        except:
            point = 9

        return point



    def processed_image(self, img):
        """
        Convert img to HSV. Get the image processed. Get 3 lines from the image.

        :parameters: input image 640x480
        :return: x, y, z: 3 coordinates
        """
        #ic("F1DQNCameraEnv.processed_image()")
        #print(f"\n F1QlearnCameraEnv.processed_image()\n")

        img_sliced = img[240:]
        img_proc = cv2.cvtColor(img_sliced, cv2.COLOR_BGR2HSV)
        line_pre_proc = cv2.inRange(img_proc, (0, 30, 30), (0, 255, 255))  # default: 0, 30, 30 - 0, 255, 200
        # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(line_pre_proc, 240, 255, cv2.THRESH_BINARY)

        lines = [mask[x_row[idx], :] for idx, x in enumerate(x_row)]
        centrals = list(map(self.get_center, lines))

        # if centrals[-1] == 9:
        #     centrals[-1] = center_image

        '''
        if telemetry_mask:
            mask_points = np.zeros((height, width), dtype=np.uint8)
            for idx, point in enumerate(centrals):
                # mask_points[x_row[idx], centrals[idx]] = 255
                cv2.line(mask_points, (point, x_row[idx]), (point, x_row[idx]), (255, 255, 255), thickness=3)

            cv2.imshow("MASK", mask_points[240:])
            cv2.waitKey(3)
        '''

        return centrals


    def step(self, action):
        #print(f"\n F1QlearnCameraEnv.step()\n")
        #ic("F1DQNCameraEnv.step()")    
        self._gazebo_unpause()

        vel_cmd = Twist()
        vel_cmd.linear.x = self.actions[action][0]
        vel_cmd.angular.z = self.actions[action][1]
        self.vel_pub.publish(vel_cmd)

        # Get camera info
        image_data = None
        f1_image_camera = None
        while image_data is None:
            image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=5)
            # Transform the image data from ROS to CVMat
            cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            f1_image_camera = self.image_msg_to_image(image_data, cv_image)
        # image_data = rospy.wait_for_message('/F1ROS/cameraL/image_raw', Image, timeout=1)
        # cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        # f1_image_camera = self.image_msg_to_image(image_data, cv_image)

        self._gazebo_pause()

        points = self.processed_image(f1_image_camera.data)
        #points = self.processed_image(cv_image)
        center = float(center_image - points[0]) / (float(width) // 2)
        center = abs(center)

        #state = self.calculate_observation(points)
        state = np.array(cv_image)

        done = False

        if center > 0.9:
            done = True
        if not done:
            if 0 <= center <= 0.2:
                reward = self.rewards['from_0_02']
            elif 0.2 < center <= 0.4:
                reward = self.rewards['from_02_04']
            else:
                reward = self.rewards['from_others']
        else:
            reward = self.rewards['from_done']

        if self.telemetry:
            print(f"\n F1DQNCameraEnv.step() -> center: {center}"
            f" - actions: {action} - reward: {reward}")
            # self.show_telemetry(f1_image_camera.data, points, action, reward)

        return state, reward, done, {}


