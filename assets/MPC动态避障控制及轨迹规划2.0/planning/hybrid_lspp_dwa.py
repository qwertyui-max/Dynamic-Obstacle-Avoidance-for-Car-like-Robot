import numpy as np
from utils.geometry import normalize_angle
import time

class HybridLSPP_DWA:
    """
    LSPP + DWA 混合控制器
    高频运行 LSPP 保证跟踪效率，低频用 DWA 验证安全性
    """
    
    def __init__(self, lspp_config, dwa_config):
        # LSPP 参数
        self.max_speed = lspp_config['max_speed']
        self.max_yawrate = lspp_config['max_yawrate']
        self.robot_radius = lspp_config['robot_radius']
        self.k1 = lspp_config.get('k1', 2.0)
        self.k2 = lspp_config.get('k2', 3.0)
        self.k3 = lspp_config.get('k3', 1.0)
        
        # DWA 安全验证参数
        self.v_samples = dwa_config['v_samples']
        self.w_samples = dwa_config['w_samples']
        self.predict_time = dwa_config['predict_time']
        self.safety_margin = dwa_config['safety_margin']
        self.emergency_stop_dist = dwa_config['emergency_stop_dist']
        
        # 状态变量
        self.step_count = 0
        self.dwa_frequency = 10  # 每10步运行一次DWA验证
        self.last_safe_v = 0.0
        self.last_safe_w = 0.0
        self.emergency_mode = False
        self.emergency_cooldown = 0
        
        # 李雅普诺夫监控
        self.lyapunov_history = []
        self.prev_lyapunov = 0.0
        self.instability_count = 0
        
        # 性能统计
        self.lspp_time = 0.0
        self.dwa_time = 0.0
        self.dwa_calls = 0
        
    def compute_lspp_control(self, state, path):
        """
        LSPP 核心控制器 - 基于李雅普诺夫稳定性理论
        """
        current_pos = state[:2]
        current_yaw = state[2]
        current_v = state[3]
        
        # 找到路径上最近的点
        distances = [np.linalg.norm(np.array(p) - current_pos) for p in path]
        closest_idx = np.argmin(distances)
        
        # 获取目标点（下一个路径点）
        target_idx = min(closest_idx + 1, len(path) - 1)
        target_point = np.array(path[target_idx])
        
        # 计算期望朝向
        dx = target_point[0] - current_pos[0]
        dy = target_point[1] - current_pos[1]
        desired_yaw = np.arctan2(dy, dx)
        
        # 计算朝向误差
        yaw_error = normalize_angle(desired_yaw - current_yaw)
        
        # 计算机器人体坐标系下的横向误差
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        target_body_x = dx * cos_yaw + dy * sin_yaw
        target_body_y = -dx * sin_yaw + dy * cos_yaw
        lateral_error = target_body_y
        
        # 计算距离终点的距离
        dist_to_goal = np.linalg.norm(current_pos - np.array(path[-1]))
        
        # ========== LSPP 速度控制 ==========
        if dist_to_goal < 2.0:
            # 终点减速
            desired_v = self.max_speed * 0.3 * (dist_to_goal / 2.0)
            desired_v = max(0.2, desired_v)
        else:
            # 基于误差调整速度
            if abs(yaw_error) > 1.0:
                desired_v = self.max_speed * 0.3
            elif abs(yaw_error) > 0.5:
                desired_v = self.max_speed * 0.5
            else:
                desired_v = self.max_speed * 0.8
        
        # ========== LSPP 角速度控制（李雅普诺夫推导） ==========
        # V = 1/2 * (k1 * e_lat^2 + e_yaw^2)
        # V_dot = -k2 * e_yaw^2  (负定，保证稳定)
        desired_w = self.k2 * yaw_error + self.k1 * lateral_error * abs(current_v)
        
        # 限制角速度
        desired_w = np.clip(desired_w, -self.max_yawrate, self.max_yawrate)
        
        # ========== 李雅普诺夫监控 ==========
        lyapunov = 0.5 * (self.k1 * lateral_error**2 + yaw_error**2)
        self.lyapunov_history.append(lyapunov)
        if len(self.lyapunov_history) > 100:
            self.lyapunov_history.pop(0)
        
        # 检查是否发散
        if len(self.lyapunov_history) > 10:
            if lyapunov > self.prev_lyapunov * 1.5 and lyapunov > 0.5:
                self.instability_count += 1
            else:
                self.instability_count = max(0, self.instability_count - 1)
        self.prev_lyapunov = lyapunov
        
        return desired_v, desired_w, lyapunov, lateral_error, yaw_error
    
    def predict_trajectory(self, state, v, w, predict_time, dt=0.1):
        """预测轨迹"""
        traj = [state[:2]]  # 只保存位置
        x, y, yaw = state[0], state[1], state[2]
        t = 0
        
        while t <= predict_time:
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
            yaw += w * dt
            traj.append([x, y])
            t += dt
        
        return np.array(traj)
    
    def check_trajectory_safety(self, traj, obstacles):
        """检查轨迹是否安全"""
        if obstacles is None or len(obstacles) == 0:
            return 1.0, None
        
        min_dist = float('inf')
        closest_obs = None
        
        for point in traj:
            for obs in obstacles:
                obs_pos = obs[:2]
                obs_radius = obs[2]  # 已包含机器人半径
                
                dist = np.linalg.norm(point - obs_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_obs = obs
                
                if dist < obs_radius:
                    return 0.0, closest_obs  # 碰撞
        
        # 计算安全分数
        safe_dist = self.robot_radius + self.safety_margin
        if min_dist < safe_dist:
            safety_score = (min_dist - 0.1) / (safe_dist - 0.1)
            safety_score = max(0.0, min(1.0, safety_score))
        else:
            safety_score = 1.0
        
        return safety_score, closest_obs
    
    def dwa_safety_verification(self, state, v_lspp, w_lspp, obstacles):
        """
        用DWA验证LSPP命令的安全性，必要时生成安全替代命令
        """
        if obstacles is None or len(obstacles) == 0:
            return True, v_lspp, w_lspp, 1.0
        
        # 预测LSPP轨迹
        lspp_traj = self.predict_trajectory(state, v_lspp, w_lspp, self.predict_time)
        
        # 检查LSPP轨迹安全性
        safety_score, _ = self.check_trajectory_safety(lspp_traj, obstacles)
        
        # 如果足够安全，直接返回
        if safety_score > 0.8:
            return True, v_lspp, w_lspp, safety_score
        
        # 如果不安全，用DWA搜索安全命令
        self.dwa_calls += 1
        start_time = time.time()
        
        # 速度采样范围
        v_min = max(-0.5, state[3] - 1.0)
        v_max = min(self.max_speed, state[3] + 1.0)
        v_samples = np.linspace(v_min, v_max, self.v_samples)
        
        w_min = max(-self.max_yawrate, state[4] - 2.0)
        w_max = min(self.max_yawrate, state[4] + 2.0)
        w_samples = np.linspace(w_min, w_max, self.w_samples)
        
        best_v, best_w = v_lspp, w_lspp
        best_score = -float('inf')
        
        for v in v_samples:
            for w in w_samples:
                # 预测轨迹
                traj = self.predict_trajectory(state, v, w, self.predict_time)
                
                # 安全检查
                safety, _ = self.check_trajectory_safety(traj, obstacles)
                
                if safety == 0.0:
                    continue
                
                # 综合评分：安全性和接近LSPP命令
                score = safety * 10.0 - 0.5 * abs(v - v_lspp) - 0.3 * abs(w - w_lspp)
                
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
        
        self.dwa_time += (time.time() - start_time)
        
        return False, best_v, best_w, safety_score
    
    def compute_avoidance_force(self, state, obstacles):
        """计算避障势能力（用于紧急情况）"""
        pos = state[:2]
        force = np.array([0.0, 0.0])
        
        for obs in obstacles:
            obs_pos = obs[:2]
            obs_radius = obs[2]
            
            dist = np.linalg.norm(pos - obs_pos)
            if dist < obs_radius + 1.0:
                # 排斥方向
                direction = (pos - obs_pos) / max(dist, 0.01)
                magnitude = max(0, (obs_radius + 1.0 - dist) / 1.0)
                force += direction * magnitude * 2.0
        
        return force
    
    def plan(self, state, global_path, obstacles):
        """
        主规划函数
        高频LSPP + 低频DWA安全验证
        """
        self.step_count += 1
        
        # 1. 计算LSPP控制命令
        lspp_start = time.time()
        v_lspp, w_lspp, lyapunov, lat_err, yaw_err = self.compute_lspp_control(state, global_path)
        self.lspp_time += (time.time() - lspp_start)
        
        # 2. 检查是否需要紧急避障
        emergency = False
        if obstacles is not None:
            # 计算到最近障碍物的距离
            min_obs_dist = float('inf')
            for obs in obstacles:
                dist = np.linalg.norm(state[:2] - obs[:2]) - obs[2]
                min_obs_dist = min(min_obs_dist, dist)
            
            if min_obs_dist < self.emergency_stop_dist:
                emergency = True
                self.emergency_mode = True
                self.emergency_cooldown = 20
        
        # 3. 紧急模式处理
        if self.emergency_mode:
            self.emergency_cooldown -= 1
            if self.emergency_cooldown <= 0:
                self.emergency_mode = False
            
            # 使用势场法避障
            force = self.compute_avoidance_force(state, obstacles)
            if np.linalg.norm(force) > 0.1:
                force_angle = np.arctan2(force[1], force[0])
                yaw_error_avoid = normalize_angle(force_angle - state[2])
                w_avoid = 3.0 * yaw_error_avoid
                v_avoid = max(0.2, v_lspp * 0.5)
                return v_avoid, w_avoid, True, lyapunov
        
        # 4. 正常模式：低频DWA安全验证
        safe = True
        final_v, final_w = v_lspp, w_lspp
        safety_score = 1.0
        
        if self.step_count % self.dwa_frequency == 0:
            safe, final_v, final_w, safety_score = self.dwa_safety_verification(
                state, v_lspp, w_lspp, obstacles
            )
            
            if not safe:
                print(f"DWA override: safety={safety_score:.2f}, "
                      f"lspp=({v_lspp:.2f},{w_lspp:.2f}) -> dwa=({final_v:.2f},{final_w:.2f})")
        
        return final_v, final_w, safe, lyapunov
    
    def get_stats(self):
        """获取性能统计"""
        total_time = self.lspp_time + self.dwa_time
        return {
            'lspp_time': self.lspp_time,
            'dwa_time': self.dwa_time,
            'dwa_calls': self.dwa_calls,
            'instability_count': self.instability_count,
            'avg_lspp_time': self.lspp_time / max(1, self.step_count),
            'avg_dwa_time': self.dwa_time / max(1, self.dwa_calls),
        }