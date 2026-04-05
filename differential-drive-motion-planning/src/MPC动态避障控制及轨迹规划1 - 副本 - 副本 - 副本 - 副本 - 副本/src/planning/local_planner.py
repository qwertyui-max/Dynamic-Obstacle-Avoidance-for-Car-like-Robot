import numpy as np
from utils.geometry import normalize_angle

def pure_pursuit_controller(current_state, global_path, config):
    """
    Pure Pursuit local path planner with improved stability
    Returns linear and angular velocity commands
    """
    if global_path is None or len(global_path) < 2:
        return 0.0, 0.0
    
    # Extract parameters with safety defaults
    lookahead_distance = config.get('lookahead_distance', 1.0)   # 基础前瞻距离
    max_lookahead = config.get('max_lookahead', 2.0)         # 最大前瞻距离
    min_lookahead = config.get('min_lookahead', 0.8)         # 最小前瞻距离
    max_speed = config.get('max_speed', 2.0)
    max_yawrate = config.get('max_yawrate', 5.0)
    
    # Current state
    current_pos = current_state[:2]
    current_yaw = current_state[2]
    current_speed = abs(current_state[3])
    
    # Adaptive lookahead based on speed and curvature(速度越快，前瞻距离越大)
    lookahead = np.clip(lookahead_distance + current_speed * 0.3, 
                        min_lookahead, max_lookahead)
    
    # Find closest point on path
    distances = [np.linalg.norm(np.array(p) - current_pos) for p in global_path]  # 最近距离
    closest_idx = np.argmin(distances)   # 路径上最近的点的索引
    
    # If we're very close to the goal
    dist_to_goal = np.linalg.norm(current_pos - np.array(global_path[-1]))
    if dist_to_goal < 0.5:
        return max_speed * 0.3, 0.0
    
    # Find lookahead point
    lookahead_point = None
    accumulated_dist = 0.0
    # 从最近点开始，沿着路径累积距离，直到达到前瞻距离
    for i in range(closest_idx, len(global_path) - 1):
        seg_start = np.array(global_path[i])
        seg_end = np.array(global_path[i + 1])
        seg_vector = seg_end - seg_start
        seg_length = np.linalg.norm(seg_vector)
        
        if seg_length < 0.01:  # Skip zero-length segments
            continue
            
        if accumulated_dist + seg_length > lookahead:
            # 在路径段上插值找到精确的前瞻点
            remaining = lookahead - accumulated_dist
            ratio = remaining / seg_length
            lookahead_point = seg_start + ratio * seg_vector
            break
        
        accumulated_dist += seg_length
    
    # If we didn't find a lookahead point, use the goal
    if lookahead_point is None:
        lookahead_point = np.array(global_path[-1])
    
    # Calculate desired heading to lookahead point
    dx = lookahead_point[0] - current_pos[0]
    dy = lookahead_point[1] - current_pos[1]
    desired_yaw = np.arctan2(dy, dx)
    
    # Calculate yaw error
    yaw_error = normalize_angle(desired_yaw - current_yaw)
    
    # Calculate curvature-based angular velocity
    # Pure pursuit formula: w = 2 * v * sin(alpha) / L
    # where alpha is angle to lookahead point, L is lookahead distance
    alpha = yaw_error
    
    # Avoid division by zero
    if lookahead < 0.1:
        lookahead = 0.1
    
    # Calculate desired angular velocity(计算控制指令）
    if abs(alpha) > 0.01:
        # Base angular velocity on pure pursuit
        desired_w = 2.0 * current_speed * np.sin(alpha) / lookahead
    else:
        desired_w = 0.0
    
    # Limit angular velocity
    desired_w = np.clip(desired_w, -max_yawrate, max_yawrate)
    
    # Adaptive linear velocity based on yaw error
    if abs(yaw_error) > 0.5:  # Sharp turn
        desired_v = max_speed * 0.2
    elif abs(yaw_error) > 0.2:  # Moderate turn
        desired_v = max_speed * 0.4
    else:  # Straight line
        desired_v = max_speed * 0.8
    
    # Slow down near goal
    if dist_to_goal < 2.0:
        desired_v = min(desired_v, max_speed * dist_to_goal / 2.0)
    
    # Ensure minimum speed for turning
    if abs(desired_w) > 0.5 and desired_v < 0.1:
        desired_v = 0.2
    
    return float(desired_v), float(desired_w)
# 计算路径上每个点的曲率
def calculate_path_curvature(path_points):
    """Calculate curvature of path at each point"""
    if len(path_points) < 3:
        return np.zeros(len(path_points))
    
    curvatures = []
    for i in range(len(path_points)):
        if i == 0 or i == len(path_points) - 1:
            curvatures.append(0)
            continue
            
        p1 = np.array(path_points[i-1])
        p2 = np.array(path_points[i])
        p3 = np.array(path_points[i+1])
        
        # Calculate curvature using Menger curvature formula
        try:
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Lengths
            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            len3 = np.linalg.norm(p3 - p1)
            
            if len1 * len2 * len3 > 0:
                # Area of triangle
                area = 0.5 * abs(np.cross(v1, v2))
                # Curvature = 4 * area / (len1 * len2 * len3)
                curvature = 4 * area / (len1 * len2 * len3)
            else:
                curvature = 0
                
            curvatures.append(min(curvature, 5.0))  # Cap curvature
            
        except:
            curvatures.append(0)
    
    return np.array(curvatures)
# 速度平滑函数
def smooth_path_velocity(path_points, max_speed, min_speed=0.1):
    """Smooth velocity profile based on path curvature"""
    if len(path_points) < 2:
        return np.ones(len(path_points)) * max_speed * 0.5
    
    curvatures = calculate_path_curvature(path_points)
    
    # Normalize curvatures with capping
    max_curve = max(np.max(curvatures), 0.1)
    norm_curvatures = np.clip(curvatures / max_curve, 0, 1)
    
    # Calculate desired velocities (inverse relationship with curvature)
    velocities = max_speed * (0.8 - 0.6 * norm_curvatures)    # 曲率越大，速度越小
    velocities = np.clip(velocities, min_speed, max_speed)
    
    # Apply smoothing filter
    try:
        from scipy.ndimage import gaussian_filter1d
        velocities = gaussian_filter1d(velocities, sigma=1.0)   # 高斯平滑处理
    except ImportError:
        # Simple moving average if scipy not available
        window = 3
        weights = np.ones(window) / window
        velocities = np.convolve(velocities, weights, mode='same')
    except:
        pass
    
    return velocities