import pybullet as p
import pybullet_data
import yaml
import collections
import ray 
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

# Constants for camera configurations
CAMERA_A = {
    'camera_distance': 2.0,
    'camera_yaw': 0.0,
    'camera_pitch': -4.7,
    'camera_target_position': [0, 0, 1]
}

CAMERA_B = {
    'camera_distance': 2.0,
    'camera_yaw': 90,
    'camera_pitch': -5.62,
    'camera_target_position': [0, 0, 1]
}

CAMERA_C = {
    'camera_distance': 2.0,
    'camera_yaw': 90,
    'camera_pitch': -89,
    'camera_target_position': [0, 0, 1]
}

cameras = [CAMERA_A, CAMERA_B, CAMERA_C]
image_keys = ['images/cam_0', 'images/cam_1', 'images/cam_2']

class PybulletEnv:
    def __init__(self, orb_position, mode='gui'):
        robot_urdf_path = "/home/vrushank/Documents/GitHub/Deep-Learning-Projects/reacher-pybullet-imitation-learning/reachy.urdf" 
        target_orientation = p.getQuaternionFromEuler([0, 0, 0])

        if mode == 'direct':
            p.connect(p.DIRECT)
        else:
            p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")
        self.right_arm_joint_indices = [1, 2, 3, 4, 5, 6, 7] 
        self.robot_id = p.loadURDF(robot_urdf_path, basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)
        self.orb_id = p.loadURDF("data/assets/red_orb.urdf", basePosition=orb_position)
        p.setCollisionFilterGroupMask(self.orb_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
        self.orb_position = orb_position

    def get_curr_obs(self):
        curr_obs = {
            'images': {},
            'agent_pos': self.get_state()
        }
        for i, camera in enumerate(cameras):
            rgb_image = np.array(self.get_image(camera))[:, :, :3]/255.0
            curr_img = np.moveaxis(rgb_image.astype(np.float32), -1, 0)
            curr_obs['images'][image_keys[i]] = curr_img
        return curr_obs

    def get_image(self, camera):
        cameraTargetPosition = camera['camera_target_position']
        cameraDistance = camera['camera_distance']
        cameraYaw = camera['camera_yaw']
        cameraPitch = camera['camera_pitch']
        _, _, rgbImg, _, _ = p.getCameraImage(
            width=224, 
            height=224, 
            viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=cameraTargetPosition,
                distance=cameraDistance,
                yaw=cameraYaw,
                pitch=cameraPitch,
                roll=0,
                upAxisIndex=2
            ),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=60, 
                aspect=1.0, 
                nearVal=0.1, 
                farVal=100.0
            )
        )
        return rgbImg
    
    def get_state(self):
        curr_joint_pos = []
        for joint_index in self.right_arm_joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_index)
            curr_joint_pos.append(joint_state[0])
        return np.array(curr_joint_pos)

    def disconnect(self):
        p.disconnect()

    def step(self, action):
        for i, joint_index in enumerate(self.right_arm_joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=action[i],
                force=100, 
                maxVelocity=1
            )
        p.stepSimulation()
        gripper_position = p.getLinkState(self.robot_id, 8)[0]
        eulerian_distance = ((gripper_position[0] - self.orb_position[0])**2 + (gripper_position[1] - self.orb_position[1])**2 + \
                             (gripper_position[2] - self.orb_position[2])**2)**0.5
        curr_obs = self.get_curr_obs()
        done = False if eulerian_distance > 0.1 else True
        return curr_obs, eulerian_distance, done

