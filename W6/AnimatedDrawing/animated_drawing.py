# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import ctypes
import heapq
import math
import time
from typing import Dict, List, Tuple, Optional, TypedDict, DefaultDict
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from skimage import measure
from shapely import geometry
from OpenGL import GL

from scipy.spatial import Delaunay
from animated_drawings.model.transform import Transform
from animated_drawings.model.time_manager import TimeManager
from animated_drawings.model.retargeter import Retargeter
from animated_drawings.model.joint import Joint
from animated_drawings.model.quaternions import Quaternions
from animated_drawings.model.vectors import Vectors
from animated_drawings.config import CharacterConfig, MotionConfig, RetargetConfig


class AnimatedDrawing(TypedDict):
    vertices: npt.NDArray[np.float32]
    triangles: List[npt.NDArray[np.int32]]


class AnimatedDrawingsJoint(Joint):
    """ Joints within Animated Drawings Rig."""

    def __init__(self, name: str, x: float, y: float):
        super().__init__(name=name, offset=np.array([x, y, 0]))
        self.starting_theta: float
        self.current_theta: float


class AnimatedDrawingRig(Transform):
    """ The skeletal rig used to deform the character """

    def __init__(self, char_cfg: CharacterConfig):
        """ Initializes character rig.  """
        super().__init__()

        # create dictionary populated with joints
        joints_d: Dict[str, AnimatedDrawingsJoint]
        joints_d = {joint['name']: AnimatedDrawingsJoint(joint['name'], *joint['loc']) for joint in char_cfg.skeleton}

        # assign joints within dictionary as childre of their parents
        for joint_d in char_cfg.skeleton:
            if joint_d['parent'] is None:
                continue
            joints_d[joint_d['parent']].add_child(joints_d[joint_d['name']])

        # updates joint positions to reflect local offsets from their parent joints
        def _update_positions(t: Transform):
            """ Now that kinematic parent-> child chain is formed, subtract parent world positions to get actual child offsets"""
            parent: Optional[Transform] = t.get_parent()
            if parent is not None:
                offset = np.subtract(t.get_local_position(), parent.get_world_position())
                t.set_position(offset)
            for c in t.get_children():
                _update_positions(c)
        _update_positions(joints_d['root'])

        # compute the starting rotation (CCW from +Y axis) of each joint
        for _, joint in joints_d.items():
            parent = joint.get_parent()
            if parent is None:
                joint.starting_theta = 0
                continue

            v1_xy = np.array([0.0, 1.0])
            v2 = Vectors([np.subtract(joint.get_world_position(), parent.get_world_position())])
            v2.norm()
            v2_xy: npt.NDArray[np.float32] = v2.vs[0, :2]
            theta = np.arctan2(v2_xy[1], v2_xy[0]) - np.arctan2(v1_xy[1], v1_xy[0])
            theta = np.degrees(theta)
            theta = theta % 360.0
            theta = np.where(theta < 0.0, theta + 360, theta)

            joint.starting_theta = float(theta)

        # attach root joint
        self.root_joint = joints_d['root']
        self.add_child(self.root_joint)

        # cache for later
        self.joint_count = joints_d['root'].joint_count()

        # set up buffer for visualizing vertices
        self.vertices = np.zeros([2 * (self.joint_count - 1), 6], np.float32)

    def set_global_orientations(self, bvh_frame_orientations: Dict[str, float]) -> None:
        """ Applies orientation from bvh_frame_orientation to the rig. """
        self._set_global_orientations(self.root_joint, bvh_frame_orientations)
        self._vertex_buffer_dirty_bit = True

    def get_joints_2D_positions(self) -> npt.NDArray[np.float32]:
        """ Returns array of 2D joints positions for rig.  """
        return np.array(self.root_joint.get_chain_worldspace_positions()).reshape([-1, 3])[:, :2]

    def _set_global_orientations(self, joint: AnimatedDrawingsJoint, bvh_orientations: Dict[str, float]) -> None:
        if joint.name in bvh_orientations.keys():

            theta: float = bvh_orientations[str(joint.name)] - joint.starting_theta
            theta = np.radians(theta)
            joint.current_theta = theta

            parent = joint.get_parent()
            assert isinstance(parent, AnimatedDrawingsJoint)
            if hasattr(parent, 'current_theta'):
                theta = theta - parent.current_theta

            rotation_q = Quaternions.from_angle_axis(np.array([theta]), axes=Vectors([0.0, 0.0, 1.0]))
            parent.set_rotation(rotation_q)
            parent.update_transforms()

        for c in joint.get_children():
            if isinstance(c, AnimatedDrawingsJoint):
                self._set_global_orientations(c, bvh_orientations)

 


class AnimatedDrawing(Transform, TimeManager):
    """
    The drawn character to be animated.
    An AnimatedDrawings object consists of four main parts:
    1. A 2D mesh textured with the original drawing, the 'visual' representation of the character
    2. A 2D skeletal rig
    3. An ARAP module which uses rig joint positions to deform the mesh
    4. A retargeting module which reposes the rig.

    After initializing the object, the retarger must be initialized by calling initialize_retarger_bvh().
    Afterwars, only the update() method needs to be called.
    """

    def __init__(self, char_cfg: CharacterConfig, retarget_cfg: RetargetConfig, motion_cfg: MotionConfig):
        super().__init__()

        self.char_cfg: CharacterConfig = char_cfg

        self.retarget_cfg: RetargetConfig = retarget_cfg

        self.img_dim: int = self.char_cfg.img_dim

        
         
        # load mask and pad to square
        # self.mask: npt.NDArray[np.uint8] = self._load_mask()

        # load texture and pad to square
        # self.txtr: npt.NDArray[np.uint8] = self._load_txtr()



        self.rig = AnimatedDrawingRig(self.char_cfg)

        self.add_child(self.rig)

        # perform runtime checks for character pose, modify retarget config accordingly
        self._modify_retargeting_cfg_for_character()

        self.retargeter: Retargeter
        self._initialize_retargeter_bvh(motion_cfg, retarget_cfg)

        self.update()

    def _modify_retargeting_cfg_for_character(self):
        """
        If the character is drawn in particular poses, the orientation-matching retargeting framework produce poor results.
        Therefore, the retargeter config can specify a number of runtime checks and retargeting modifications to make if those checks fail.
        """
        for position_test, target_joint_name, joint1_name, joint2_name in self.retarget_cfg.char_runtime_checks:
            if position_test == 'above':
                """ Checks whether target_joint is 'above' the vector from joint1 to joint2. If it's below, removes it.
                This was added to account for head flipping when nose was below shoulders. """

                # get joints 1, 2 and target joint
                joint1 = self.rig.root_joint.get_transform_by_name(joint1_name)
                if joint1 is None:
                    msg = f'Could not find joint1 in runtime check: {joint1_name}'
                    logging.critical(msg)
                    assert False, msg
                joint2 = self.rig.root_joint.get_transform_by_name(joint2_name)
                if joint2 is None:
                    msg = f'Could not find joint2 in runtime check: {joint2_name}'
                    logging.critical(msg)
                    assert False, msg
                target_joint = self.rig.root_joint.get_transform_by_name(target_joint_name)
                if target_joint is None:
                    msg = f'Could not find target_joint in runtime check: {target_joint_name}'
                    logging.critical(msg)
                    assert False, msg

                # get world positions
                joint1_xyz = joint1.get_world_position()
                joint2_xyz = joint2.get_world_position()
                target_joint_xyz = target_joint.get_world_position()

                # rotate target vector by inverse of test_vector angle. If then below x axis discard it.
                test_vector = np.subtract(joint2_xyz, joint1_xyz)
                target_vector = np.subtract(target_joint_xyz, joint1_xyz)
                angle = math.atan2(test_vector[1], test_vector[0])
                if (math.sin(-angle) * target_vector[0] + math.cos(-angle) * target_vector[1]) < 0:
                    logging.info(f'char_runtime_check failed, removing {target_joint_name} from retargeter :{target_joint_name, position_test, joint1_name, joint2_name}')
                    del self.retarget_cfg.char_joint_bvh_joints_mapping[target_joint_name]
            else:
                msg = f'Unrecognized char_runtime_checks position_test: {position_test}'
                logging.critical(msg)
                assert False, msg

    def _initialize_retargeter_bvh(self, motion_cfg: MotionConfig, retarget_cfg: RetargetConfig):
        """ Initializes the retargeter used to drive the animated character.  """

        # initialize retargeter
        self.retargeter = Retargeter(motion_cfg, retarget_cfg)

        # validate the motion and retarget config files, now that we know char/bvh joint names
        char_joint_names: List[str] = self.rig.root_joint.get_chain_joint_names()
        bvh_joint_names = self.retargeter.bvh_joint_names
        motion_cfg.validate_bvh(bvh_joint_names)
        retarget_cfg.validate_char_and_bvh_joint_names(char_joint_names, bvh_joint_names)

        # a shorter alias
        char_bvh_root_offset: RetargetConfig.CharBvhRootOffset = self.retarget_cfg.char_bvh_root_offset

        # compute ratio of character's leg length to bvh skel leg length
        c_limb_length = 0
        c_joint_groups: List[List[str]] = char_bvh_root_offset['char_joints']
        for b_joint_group in c_joint_groups:
            while len(b_joint_group) >= 2:
                c_dist_joint = self.rig.root_joint.get_transform_by_name(b_joint_group[1])
                c_prox_joint = self.rig.root_joint.get_transform_by_name(b_joint_group[0])
                assert isinstance(c_dist_joint, AnimatedDrawingsJoint)
                assert isinstance(c_prox_joint, AnimatedDrawingsJoint)
                c_dist_joint_pos = c_dist_joint.get_world_position()
                c_prox_joint_pos = c_prox_joint.get_world_position()
                c_limb_length += np.linalg.norm(np.subtract(c_dist_joint_pos, c_prox_joint_pos))
                b_joint_group.pop(0)

        b_limb_length = 0
        b_joint_groups: List[List[str]] = char_bvh_root_offset['bvh_joints']
        for b_joint_group in b_joint_groups:
            while len(b_joint_group) >= 2:
                b_dist_joint = self.retargeter.bvh.root_joint.get_transform_by_name(b_joint_group[1])
                b_prox_joint = self.retargeter.bvh.root_joint.get_transform_by_name(b_joint_group[0])
                assert isinstance(b_dist_joint, Joint)
                assert isinstance(b_prox_joint, Joint)
                b_dist_joint_pos = b_dist_joint.get_world_position()
                b_prox_joint_pos = b_prox_joint.get_world_position()
                b_limb_length += np.linalg.norm(np.subtract(b_dist_joint_pos, b_prox_joint_pos))
                b_joint_group.pop(0)

        # compute character-bvh scale factor and send to retargeter
        scale_factor = float(c_limb_length / b_limb_length)
        projection_bodypart_group_for_offset = char_bvh_root_offset['bvh_projection_bodypart_group_for_offset']
        self.retargeter.scale_root_positions_for_character(scale_factor, projection_bodypart_group_for_offset)

        # compute the necessary orienations
        for char_joint_name, (bvh_prox_joint_name, bvh_dist_joint_name) in self.retarget_cfg.char_joint_bvh_joints_mapping.items():
            self.retargeter.compute_orientations(bvh_prox_joint_name, bvh_dist_joint_name, char_joint_name)

    def update(self):
        """
        This method receives the delta t, the amount of time to progress the character's internal time keeper.
        This method passes its time to the retargeter, which returns bone orientations.
        Orientations are passed to rig to calculate new joint positions.
        The updated joint positions are passed into the ARAP module, which computes the new vertex locations.
        The new vertex locations are stored and the dirty bit is set.
        """

        # get retargeted motion data
        frame_orientations: Dict[str, float]
        joint_depths: Dict[str, float]
        root_position: npt.NDArray[np.float32]
        frame_orientations, joint_depths, root_position = self.retargeter.get_retargeted_frame_data(self.get_time())

        # update the rig's root position and reorient all of its joints
        self.rig.root_joint.set_position(root_position)
        self.rig.set_global_orientations(frame_orientations)

        # using new joint positions, calculate new mesh vertex xy positions
        control_points: npt.NDArray[np.float32] = self.rig.get_joints_2D_positions() - root_position[:2]
        # self.vertices[:, :2] = self.arap.solve(control_points) + root_position[:2]

        # use the z position of the rig's root joint for all mesh vertices
        # self.vertices[:, 2] = self.rig.root_joint.get_world_position()[2]

        # self._vertex_buffer_dirty_bit = True

    def getUpdatedJointPositions(self, printout=False):
        """
        This method receives the delta t, the amount of time to progress the character's internal time keeper.
        This method passes its time to the retargeter, which returns bone orientations.
        Orientations are passed to rig to calculate new joint positions.
        The updated joint positions are passed into the ARAP module, which computes the new vertex locations.
        The new vertex locations are stored and the dirty bit is set.
        """

        # get retargeted motion data
        frame_orientations: Dict[str, float]
        joint_depths: Dict[str, float]
        root_position: npt.NDArray[np.float32]
        frame_orientations, joint_depths, root_position = self.retargeter.get_retargeted_frame_data(self.get_time())

        # update the rig's root position and reorient all of its joints
        self.rig.root_joint.set_position(root_position)
        self.rig.set_global_orientations(frame_orientations)

        # using new joint positions, calculate new mesh vertex xy positions
        control_points: npt.NDArray[np.float32] = self.rig.get_joints_2D_positions() - root_position[:2]
        if printout:
            print(control_points.shape)
            print(control_points)
  

        return control_points, root_position[:2]

  