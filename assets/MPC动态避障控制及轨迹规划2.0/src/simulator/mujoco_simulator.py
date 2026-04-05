import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def get_car_state(data):
    x = data.qpos[0]
    y = data.qpos[1]
    quat = data.qpos[3:7]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    yaw = rot.as_euler('zyx')[0]
    vx_global = data.qvel[0]
    vy_global = data.qvel[1]
    vx_body = vx_global * np.cos(yaw) + vy_global * np.sin(yaw)
    wz = data.qvel[5]
    return np.array([x, y, yaw, vx_body, wz])

def extract_obstacles(model):
    """
    正确提取障碍物 - 对于长方体，使用对角线的一半作为半径
    XML中障碍物是长方体 size="0.05 0.05 0.5"，所以：
    - 半长: 0.05, 0.05
    - 对角线半长 = sqrt(0.05^2 + 0.05^2) = 0.0707
    - 为了安全避障，应该使用更大的半径
    """
    obstacles = []
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "obstacle" in geom_name:
            pos = model.geom_pos[i]
            size = model.geom_size[i]
            
            # 对于长方体，计算对角线长度的一半作为半径
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                # size是半长，对角线半长 = sqrt(size[0]^2 + size[1]^2)
                radius = math.sqrt(size[0]**2 + size[1]**2)
                print(f"Box obstacle: size={size}, radius={radius:.3f}")
            else:
                radius = math.sqrt(size[0]**2 + size[1]**2)
            
            obstacles.append([pos[0], pos[1], radius])
    
    return np.array(obstacles)

def check_collision(model, data):
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        
        if geom1_name is None or geom2_name is None:
            continue
            
        if ("chasis" in geom1_name and "obstacle" in geom2_name) or \
           ("chasis" in geom2_name and "obstacle" in geom1_name):
            return True
    return False