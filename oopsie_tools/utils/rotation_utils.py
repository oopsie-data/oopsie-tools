from scipy.spatial.transform import Rotation as R
import numpy as np
from enum import Enum


class RotOption(Enum):
    XYZ = 0
    ZYX = 1
    XYX = 2
    QUAT = 3
    MATRIX = 4
    ROT6D = 5
    ROTVEC = 6
    xyz = 7
    zyx = 8
    xyx = 9

    @staticmethod
    def from_string(s: str):
        if s.startswith("euler_"):
            format = s[len("euler_") :]
            if format in ["xyz", "zyx", "xyx", "XYZ", "ZYX", "XYX"]:
                return RotOption[format]
            else:
                raise ValueError(f"Unsupported Euler format: {format}")
        elif s == "quat":
            return RotOption.QUAT
        elif s == "matrix":
            return RotOption.MATRIX
        elif s == "rot6d":
            return RotOption.ROT6D
        elif s == "rotvec":
            return RotOption.ROTVEC
        else:
            raise ValueError(f"Unsupported rotation option: {s}")


class ActionQuatConversion:
    def __init__(self, source_format, is_biarm=False):
        self.rot_option = source_format
        self.is_biarm = is_biarm

    def _to_quat(self, arr):
        if self.rot_option == RotOption.QUAT:
            return arr
        elif self.rot_option in [RotOption.XYZ, RotOption.ZYX, RotOption.XYX, RotOption.xyz, RotOption.zyx, RotOption.xyx]:
            r = R.from_euler(self.rot_option.name, arr)
            return r.as_quat()  # returns in (x, y, z, w) order
        elif self.rot_option == RotOption.MATRIX:
            r = R.from_matrix(arr)
            return r.as_quat()
        elif self.rot_option == RotOption.ROT6D:
            # Convert 6D rotation representation to rotation matrix
            rot6d = arr.reshape(2, 3)
            u1 = rot6d[0] / np.linalg.norm(rot6d[0])
            u2 = rot6d[1] - np.dot(rot6d[1], u1) * u1
            u2 /= np.linalg.norm(u2)
            u3 = np.cross(u1, u2)
            rot_matrix = np.stack([u1, u2, u3], axis=1)
            r = R.from_matrix(rot_matrix)
            return r.as_quat()
        elif self.rot_option == RotOption.ROTVEC:
            r = R.from_rotvec(arr)
            return r.as_quat()
        else:
            raise ValueError(f"Unsupported rotation option: {self.rot_option}")

    def _convert_arm(self, action):
        arm_rot = self._to_quat(action[3:])
        return np.concatenate([action[:3], arm_rot])  # position + converted rotation

    def convert_position(self, action):
        action_dim = len(action)
        if self.is_biarm:
            arm1_action = self._convert_arm(action[: action_dim // 2])
            arm2_action = self._convert_arm(action[action_dim // 2 :])
            return np.concatenate([arm1_action, arm2_action])
        else:
            return self._convert_arm(action)
