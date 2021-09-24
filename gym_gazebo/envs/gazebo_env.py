import gym
import rospy
#import roslaunch
import sys
import os
import signal
from cprint import cprint
from icecream import ic
from datetime import datetime, timedelta
import numpy as np


from pathlib import Path

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState

import subprocess
from std_srvs.srv import Empty
import random
from rosgraph_msgs.msg import Clock



ic.enable()
#ic.disable()
#ic.configureOutput(prefix='Debug | ')
ic.configureOutput(prefix=f'{datetime.now()} | ')


class GazeboEnv(gym.Env):
    """
    Superclass for all Gazebo environments.
    """
    
    metadata = {'render.models': ['human']}

    #def __init__(self, launchfile):
    def __init__(self, **config):

        #cprint.ok(f"\n -------- Enter in GazeboEnv -----------\n")
        self.launchfile = config.get("launch")
        #cprint.info(f"[GazeboEnv] -> launchfile: {self.launchfile}")
        self.agent = config.get("agent")
        self.last_clock_msg = Clock()
        #self.port = "11311"  # str(random_number) #os.environ["ROS_PORT_SIM"]
        #self.port_gazebo = "11345"  # str(random_number+1) #os.environ["ROS_PORT_SIM"]
        self.port = config['ROS_MASTER_URI']  # str(random_number) #os.environ["ROS_PORT_SIM"]
        self.port_gazebo = config['GAZEBO_MASTER_URI']        
        # self.ros_master_uri = os.environ["ROS_MASTER_URI"];
        # self.port = os.environ.get("ROS_PORT_SIM", "11311")

        #print(f"\n[GazeboEnv] -> ROS_MASTER_URI = http://localhost:{self.port}\n")
        #print(f"\n[GazeboEnv] -> GAZEBO_MASTER_URI = http://localhost:{self.port_gazebo}\n")

        ros_path = os.path.dirname(subprocess.check_output(["which", "roscore"]))
        #cprint.warn(f"\n[GazeboEnv] -> ros_path: {ros_path}")

        # NOTE: It doesn't make sense to launch a roscore because it will be done when spawing Gazebo, which also need
        #   to be the first node in order to initialize the clock.
        # # start roscore with same python version as current script
        # self._roscore = subprocess.Popen([sys.executable, os.path.join(ros_path, b"roscore"), "-p", self.port])
        # time.sleep(1)
        # print ("Roscore launched!")

        if self.launchfile.startswith("/"):
            fullpath = self.launchfile
        else:
            # TODO: Global env for 'f1'. It must be passed in constructor.
            fullpath = str(Path(Path(__file__).resolve().parents[1] / "CustomRobots" / self.agent / "launch" / self.launchfile))
            #print(f"\n[GazeboEnv] -> fullpath: {fullpath}")
        if not os.path.exists(fullpath):
            raise IOError(f"[GazeboEnv] -> File {fullpath} does not exist")


        # Launching Gazebo only first time
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        #roscore_count = tmp.count('roscore')
        #rosmaster_count = tmp.count('rosmaster')

        if gzclient_count == 0 and gzserver_count == 0:
            #ic("entro primera vez")
            self._roslaunch = subprocess.Popen([
                sys.executable, os.path.join(ros_path, b"roslaunch"), "-p", self.port, fullpath
            ])


        #ic(self._roslaunch)
        #print("\n[GazeboEnv] -> Gazebo launched!")
        #ic("GAZEBO LAUNCHED")

        self.gzclient_pid = 0

        # Launch the simulation with the given launchfile name
        rospy.init_node('gym', anonymous=True)

        #cprint.ok(f"\n [GazeboEnv] -> -------- Out GazeboEnv (__init__) ----------------\n")


    def _gazebo_reset(self):
        # Resets the state of the environment and returns an initial observation.
        #print(f"\n GazeboEnv._gazebo_reset()\n")
        rospy.wait_for_service('/gazebo/reset_simulation')

        #rospy.wait_for_service('/gzserver/reset_simulation')
        #rospy.wait_for_service('/gazebo_ros/gzserver/reset_simulation')
        #print("pase")
        try:
            # reset_proxy.call()
            self.reset_proxy()
            self.unpause()
            self.start_pose
            self.position = None

            #rospy.ServiceProxy('/F1ROS/cameraL/image_raw', Empty)
            ic(self.start_pose)
            ic(self.position)
        except rospy.ServiceException as e:
            print(f"/gazebo/reset_simulation service call failed: {e}")


    def _gazebo_set_new_pose(self):
        """
        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        #print(f"\n GazeboEnv._gazebo_set_new_pose()\n")
        #pos = random.choice(list(enumerate(self.circuit["gaz_pos"])))[0]
        #self.position = pos

        #pos_number = self.circuit["gaz_pos"][0]

        posit = np.random.randint(5)
        pos_number = self.start_random_pose[posit][0]
        ic(pos_number)

        state = ModelState()
        state.model_name = "f1_renault"
        state.pose.position.x = self.start_random_pose[posit][1]
        state.pose.position.y = self.start_random_pose[posit][2]
        state.pose.position.z = self.start_random_pose[posit][3]
        state.pose.orientation.x = self.start_random_pose[posit][4]
        state.pose.orientation.y = self.start_random_pose[posit][5]
        state.pose.orientation.z = self.start_random_pose[posit][6]
        state.pose.orientation.w = self.start_random_pose[posit][7]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state)
            #ic(set_state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number

    def _gazebo_pause(self):
        #print(f"\n GazeboEnv._gazebo_pause()\n")
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except rospy.ServiceException as e:
            print(f"/gazebo/pause_physics service call failed: {e}")

    def _gazebo_unpause(self):
        #print(f"\n GazeboEnv._gazebo_unpause()\n")
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except rospy.ServiceException as e:
            print(f"/gazebo/unpause_physics service call failed: {e}")

    '''
       
    def _gazebo_set_new_pose(self):

        """
        This is NAcho's original

        (pos_number, pose_x, pose_y, pose_z, or_x, or_y, or_z, or_z)
        """
        print(f"\n GazeboEnv._gazebo_set_new_pose()\n")
        pos = random.choice(list(enumerate(self.circuit["gaz_pos"])))[0]
        self.position = pos

        pos_number = self.circuit["gaz_pos"][0]

        state = ModelState()
        state.model_name = "f1_renault"
        state.pose.position.x = self.circuit["gaz_pos"][pos][1]
        state.pose.position.y = self.circuit["gaz_pos"][pos][2]
        state.pose.position.z = self.circuit["gaz_pos"][pos][3]
        state.pose.orientation.x = self.circuit["gaz_pos"][pos][4]
        state.pose.orientation.y = self.circuit["gaz_pos"][pos][5]
        state.pose.orientation.z = self.circuit["gaz_pos"][pos][6]
        state.pose.orientation.w = self.circuit["gaz_pos"][pos][7]

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            set_state(state)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
        return pos_number

    '''


    def _render(self, mode="human", close=False):
        print(f"\n GazeboEnv._render()\n")
        if close:
            tmp = os.popen("ps -Af").read()
            proccount = tmp.count('gzclient')
            if proccount > 0:
                if self.gzclient_pid != 0:
                    os.kill(self.gzclient_pid, signal.SIGTERM)
                    os.wait()
            return

        tmp = os.popen("ps -Af").read()
        proccount = tmp.count('gzclient')
        if proccount < 1:
            subprocess.Popen("gzclient")
            self.gzclient_pid = int(subprocess.check_output(["pidof", "-s", "gzclient"]))
        else:
            self.gzclient_pid = 0

    @staticmethod
    def _close():
        print(f"\n GazeboEnv._close()\n")
        # Kill gzclient, gzserver and roscore
        tmp = os.popen("ps -Af").read()
        gzclient_count = tmp.count('gzclient')
        gzserver_count = tmp.count('gzserver')
        roscore_count = tmp.count('roscore')
        rosmaster_count = tmp.count('rosmaster')

        if gzclient_count > 0:
            os.system("killall -9 gzclient")
        if gzserver_count > 0:
            os.system("killall -9 gzserver")
        if rosmaster_count > 0:
            os.system("killall -9 rosmaster")
        if roscore_count > 0:
            os.system("killall -9 roscore")

        if gzclient_count or gzserver_count or roscore_count or rosmaster_count > 0:
            os.wait()

    def _configure(self):

        # TODO
        # From OpenAI API: Provides runtime configuration to the enviroment
        # Maybe set the Real Time Factor?
        pass

    def _seed(self):

        # TODO
        # From OpenAI API: Sets the seed for this env's random number generator(s)
        pass


    ''' Pedro: tratando de repolicar funcion, pero ya funciona en la otra
    
    def _gazebo_reset_random_pose(self):
        # Resets the state of the environment and returns an initial observation.
        #print(f"\n GazeboEnv._gazebo_reset()\n")
        rospy.wait_for_service('/gazebo/reset_simulation')

        #rospy.wait_for_service('/gzserver/reset_simulation')
        #rospy.wait_for_service('/gazebo_ros/gzserver/reset_simulation')
        #print("pase")
        try:
            # reset_proxy.call()
            self.reset_proxy()
            self.unpause()

            #self.start_random_pose =[]
            posit = np.random.randint(5)
            self.start_random_pose = [self.start_random_pose[posit][1], self.start_random_pose[posit][2]]
            self.position = None

            #rospy.ServiceProxy('/F1ROS/cameraL/image_raw', Empty)
            ic(self.start_random_pose)
            ic(self.position)
        except rospy.ServiceException as e:
            print(f"/gazebo/reset_simulation service call failed: {e}")

    '''

    def step(self, action):

        # Implement this method in every subclass
        # Perform a step in gazebo. E.g. move the robot
        raise NotImplementedError

    def reset(self):

        # Implemented in subclass
        raise NotImplementedError

    def get_position(self):
        object_coordinates = self.model_coordinates("f1_renault", "")
        x_position = round(object_coordinates.pose.position.x, 2)
        y_position = round(object_coordinates.pose.position.y, 2)
        print(f"\n GazeboEnv.get_position()\n")

        return x_position, y_position
