from collections import OrderedDict
import random
import numpy as np

import utils.transform_utils as T
from utils.mjcf_utils import string_to_array
from utils.transform_utils import convert_quat
from environments.panda import PandaEnv

from models.arena import TableArena
from models.objects import CubeObject, BasePartObject, CylObject, Cyl2Object
from models.robot import Panda
from models.task import GraspingTask

import hjson
import os


class PandaGrasp(PandaEnv):
    """
    This class corresponds to the grasping task for the 🐼️ robot arm.
    """

    def __init__(self, config):
        """
        Args:
            config = configuration file of environment and task parameters
        """

        # settings for table top
        self.config = config
        self.table_full_size = config.table_full_size

        # Load the controller parameter configuration files
        controller_filepath = os.path.join(os.path.dirname(__file__), '..','config/controller_config.hjson')
        super().__init__(config, controller_config_file=controller_filepath)


    def _load_model(self):
        super()._load_model()
        # set the robot base pos in world ref frame; World frame is right at the center of floor
        self.mujoco_robot.set_base_xpos([-0.5, 0, 0.913])

        # load model for table top workspace
        self.mujoco_arena = TableArena(table_full_size=self.table_full_size)
        self.mujoco_arena.set_origin([0, 0, 0])

        # define mujoco objects
        Cube = CubeObject()
        Base = BasePartObject() # This object has no function in learning its just for viewing purpose

        self.mujoco_objects = OrderedDict([("BasePart", Base), ("Cube", Cube)])
        
        # task includes arena, robot, and objects of interest
        self.model = GraspingTask(self.mujoco_arena, self.mujoco_robot, self.mujoco_objects)
        self.table_pos = string_to_array(self.model.table_body.get("pos"))


    def _get_reference(self):
        super()._get_reference()

        self.cube_body_id = self.sim.model.body_name2id("Cube") # 'cube': 23
        self.cube_geom_id = self.sim.model.geom_name2id("Cube") # 'cube': 41

        #information of objects
        self.object_names = list(self.mujoco_objects.keys())
        self.object_site_ids = [self.sim.model.site_name2id(ob_name) for ob_name in self.object_names]


        # id of grippers for contact checking
        # ['hand_collision', 'finger1_collision', 'finger2_collision', 'finger1_tip_collision', 'finger2_tip_collision']
        if self.has_gripper:
            self.finger_names = self.gripper.contact_geoms()
            self.l_finger_geom_ids = [self.sim.model.geom_name2id(x) for x in self.gripper.left_finger_geoms] # finger1_visual': 34
            self.r_finger_geom_ids = [self.sim.model.geom_name2id(x) for x in self.gripper.right_finger_geoms] # 'finger2_visual': 37

        # self.sim.data.contact # list, geom1, geom2
        self.collision_check_geom_names = self.sim.model._geom_name2id.keys() # {...,'hand_collision': 33 ,...,'BasePart': 40, 'cube': 41}
        self.collision_check_geom_ids = [self.sim.model._geom_name2id[k] for k in self.collision_check_geom_names]


    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        init_pos = np.array([0, np.pi / 16.0, 0.00, -np.pi / 2.0 - np.pi / 3.0, 0.00, np.pi - 0.2, np.pi / 4])

        # Uncomment below line for adding randomization to initial position of robot position
        #init_pos += np.random.randn(init_pos.shape[0]) * 0.02

        self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)

        # reset position of objects on table top

        # Cube
        self.sim.data.qpos[16:19] = np.array([0 ,-0.1 ,0.85])
        self.sim.data.qpos[19:23] = np.array([1 ,0 ,0, 0])

        # Random Base Part
        self.sim.data.qpos[9:12] = np.array([0 ,0.1 ,0.82])
        self.sim.data.qpos[12:16] = np.array([1 ,0 ,0, 0])

        if self.config.mode == 2:
            # reset cylinders pos
            self.sim.data.qpos[23:26] = np.array([0 ,0.03 ,0.9])
            self.sim.data.qpos[26:30] = np.array([1 ,0 ,0, 0])

            self.sim.data.qpos[30:33] = np.array([0 ,-0.23 ,0.9])
            self.sim.data.qpos[33:37] = np.array([1 ,0 ,0, 0])

        # reset grasp state
        self.has_grasp = False

    def reward(self, action=None):
        """
        Reward function for the task.
        Returns:
            reward (float): the reward
        """
        # TODO: Implement the reward to grasp and lift the cube

        # reaching reward
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]

        reward = 0

        return reward

    def _check_success(self):
        """
        Returns True if task has been completed.
        """
        cube_pos = self.sim.data.body_xpos[self.cube_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cube_pos)

        return self.has_grasp and cube_pos[2] > 0.86


    def _get_observation(self):
        """
        Returns an OrderedDict containing object observations
        """
        state = super()._get_observation()
        di = OrderedDict()

        # low-level object information
        # position and rotation of object
        cube_pos = np.array(self.sim.data.body_xpos[self.cube_body_id])
        cube_quat = convert_quat(np.array(self.sim.data.body_xquat[self.cube_body_id]), to="xyzw")
        di["cube_pos"] = cube_pos
        di["cube_quat"] = cube_quat

        gripper_site_pos = np.array(self.sim.data.site_xpos[self.eef_site_id])
        # gripper to cube distance
        di["gripper_to_cube"] = gripper_site_pos - cube_pos

        state["object-state"] = np.concatenate([cube_pos, cube_quat, di["gripper_to_cube"]])

        return state
