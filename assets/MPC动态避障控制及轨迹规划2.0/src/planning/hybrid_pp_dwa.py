import numpy as np
import time
from utils.geometry import normalize_angle

class HybridPP_DWA:
    """
    Pure Pursuit + DWA 混合控制器
    高频运行 Pure Pursuit 保证跟踪效率，低频用 DWA 验证安全性
    """

    def __init__(self, pp_config, dwa_config):
        # Pure Pursuit 参数
        self.max_speed = pp_config['max_speed']
        self.max_yawrate = pp_config['max_yawrate']
        self.robot_radius = pp_config['robot_radius']
        self.lookahead_distance = pp_config['lookahead_distance']
        self.max_lookahead = pp_config['max_lookahead']
        self.min_lookahead = pp_config['min_lookahead']
        self.speed_gain = pp_config.get('speed_gain', 0.8)
        self.turn_gain = pp_config.get('turn_gain', 1.5)

        # DWA 参数
        self.v_samples = dwa_config['v_samples']  # 速度采样数
        self.w_samples = dwa_config['w_samples']  # 角速度采样数
        self.predict_time = dwa_config['predict_time']  # 预测时间
        self.safety_margin = dwa_config['safety_margin']  # 安全裕度
        self.emergency_stop_dist = dwa_config['emergency_stop_dist']  # 紧急停车距离
        self.collision_weight = dwa_config.get('collision_weight', 15.0)  # 碰撞权重
        self.path_weight = dwa_config.get('path_weight', 2.0)  # 路径权重
        self.speed_weight = dwa_config.get('speed_weight', 0.3)  # 速度权重
        self.obstacle_cost_threshold = dwa_config.get('obstacle_cost_threshold', 0.7)  # 障碍物代价阈值

        # 状态变量
        self.step_count = 0
        self.dwa_frequency = 10  # DWA验证频率(每10步验证一次)
        self.emergency_mode = False  # 紧急模式标志
        self.emergency_cooldown = 0  # 紧急模式冷却
        self.last_collision_pos = None  # 上次碰撞位置
        self.collision_memory = []  # 碰撞记忆

        # 性能统计
        self.pp_time = 0.0  # PP总耗时
        self.dwa_time = 0.0  # DWA总耗时
        self.dwa_calls = 0  # DWA调用次数
        self.override_count = 0  # DWA覆盖次数

    def pure_pursuit_control(self, state, path):
        """
        Pure Pursuit 核心控制器
        """

        current_pos = state[:2]    # 当前位置
        current_yaw = state[2]     # 当前朝向
        current_speed = abs(state[3])  # 当前速度
        
        # 找到路径上最近的点
        distances = [np.linalg.norm(np.array(p) - current_pos) for p in path]
        closest_idx = np.argmin(distances)
        
        # 计算距离终点的距离
        dist_to_goal = np.linalg.norm(current_pos - np.array(path[-1]))
        
        # 自适应前视距离 - 根据速度和距离目标的距离
        if dist_to_goal < 2.0:
            # 接近目标时减小前视距离
            lookahead = max(0.5, dist_to_goal * 0.5)
        else:
            # 正常情况：速度越快，前视距离越大
            lookahead = np.clip(1.2 + current_speed * 0.4, 0.8, 2.2)
        
        # 找到前视点
        lookahead_point = None
        accumulated_dist = 0.0
        
        # 从最近点开始向前找
        for i in range(closest_idx, len(path) - 1):
            seg_start = np.array(path[i])      # 当前段起点
            seg_end = np.array(path[i + 1])    # 当前段终点
            seg_length = np.linalg.norm(seg_end - seg_start)   # 当前路径段的长度
            # 如果加上这段就会超过前视距离
            if accumulated_dist + seg_length > lookahead:
                remaining = lookahead - accumulated_dist  # 到达前视距离的剩余距离
                ratio = remaining / seg_length if seg_length > 0 else 0  # 计算比例
                lookahead_point = seg_start + ratio * (seg_end - seg_start)  # 插值得到精确的前视点
                break
            elif i == len(path) - 2:
                # 如果是最后一段，直接用终点
                lookahead_point = seg_end
                break
            
            accumulated_dist += seg_length
        # 如果没找到，用终点
        if lookahead_point is None:
            lookahead_point = np.array(path[-1])
        
        # 计算期望朝向
        dx = lookahead_point[0] - current_pos[0]
        dy = lookahead_point[1] - current_pos[1]
        
        # 避免除零
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return 0.0, 0.0, 0.0, lookahead_point
        
        desired_yaw = np.arctan2(dy, dx)   # 期望朝向
        
        # 计算朝向误差
        yaw_error = normalize_angle(desired_yaw - current_yaw)
        
        # Pure Pursuit 角速度公式
        if lookahead < 0.1:
            lookahead = 0.1
        
        if current_speed > 0.01:
            desired_w = 2.0 * current_speed * np.sin(yaw_error) / lookahead # Pure Pursuit公式：ω=2v*sin(α)/L
        else:
            # 低速时使用比例控制
            desired_w = self.turn_gain * yaw_error
        
        desired_w = np.clip(desired_w, -self.max_yawrate, self.max_yawrate)
        
        # 速度控制
        if dist_to_goal < 2.0:
            # 接近终点时减速
            desired_v = self.max_speed * 0.5 * (dist_to_goal / 2.0)   # 接近终点的时候减速
            desired_v = max(0.2, desired_v)
        else:
            # 根据转向误差调整速度(转向越大，速度越慢)
            if abs(yaw_error) > 0.8:
                desired_v = self.max_speed * 0.3
            elif abs(yaw_error) > 0.4:
                desired_v = self.max_speed * 0.5
            elif abs(yaw_error) > 0.2:
                desired_v = self.max_speed * 0.7
            else:
                desired_v = self.max_speed * 0.9
        
        desired_v = np.clip(desired_v * self.speed_gain, 0.2, self.max_speed)
        
        return desired_v, desired_w, yaw_error, lookahead_point
    
    def check_collision_risk(self, state, obstacles):
        """
        检查碰撞风险
        返回: 安全分数 (0-1), 最近的障碍物
        """
        if obstacles is None or len(obstacles) == 0:
            return 1.0, None
        
        pos = state[:2]
        min_dist = float('inf')
        closest_obs = None
        
        for obs in obstacles:
            obs_pos = obs[:2]
            obs_radius = obs[2]
            
            # 计算到障碍物边缘的距离
            dist = np.linalg.norm(pos - obs_pos) - obs_radius
            if dist < min_dist:
                min_dist = dist
                closest_obs = obs
        
        # 计算安全分数
        if min_dist < 0:
            # 已经在障碍物内
            return 0.0, closest_obs
        elif min_dist < self.robot_radius:
            # 在机器人半径内
            safety = min_dist / self.robot_radius
            return max(0.0, safety), closest_obs
        elif min_dist < self.robot_radius + self.safety_margin:
            # 在安全边际内
            safety = 1.0 - (min_dist - self.robot_radius) / self.safety_margin
            return max(0.0, safety), closest_obs
        else:
            # 安全
            return 1.0, None
    
    def predict_trajectory(self, state, v, w):
        """预测未来轨迹"""
        trajectory = [state[:2]]
        x, y, yaw = state[0], state[1], state[2]
        dt = 0.1
        steps = int(self.predict_time / dt)
        
        for _ in range(steps):
            x += v * np.cos(yaw) * dt
            y += v * np.sin(yaw) * dt
            yaw += w * dt
            trajectory.append([x, y])
        
        return np.array(trajectory)
    
    def compute_avoidance_command(self, state, obstacles):
        """
        计算紧急避障命令 - 使用势场法
        """
        pos = state[:2]
        yaw = state[2]
        
        # 计算障碍物合力
        force = np.array([0.0, 0.0])
        min_dist = float('inf')
        
        for obs in obstacles:
            obs_pos = obs[:2]
            obs_radius = obs[2]
            
            dist = np.linalg.norm(pos - obs_pos)
            if dist < obs_radius + 1.5:
                # 排斥方向（从障碍物指向机器人）
                direction = (pos - obs_pos) / max(dist, 0.01)
                # 排斥力大小（距离越近，力越大）
                magnitude = 1.0 - (dist - obs_radius) / 1.5
                magnitude = max(0.0, min(1.0, magnitude))
                force += direction * magnitude * 2.5
                
                if dist < min_dist:
                    min_dist = dist
        
        # 如果非常危险，紧急倒车
        if min_dist < self.emergency_stop_dist:
            return -0.5, 0.0
        
        if np.linalg.norm(force) > 0.1:
            # 转向远离障碍物的方向
            avoid_angle = np.arctan2(force[1], force[0])
            yaw_error = normalize_angle(avoid_angle - yaw)
            
            # 根据危险程度决定转向强度
            danger_level = 1.0 - min(1.0, min_dist / (self.robot_radius + self.safety_margin + 0.3))
            w_avoid = 3.5 * yaw_error * danger_level
            w_avoid = np.clip(w_avoid, -self.max_yawrate, self.max_yawrate)
            
            # 根据危险程度决定速度
            v_avoid = 0.4 * (1.0 - danger_level * 0.6)
            
            return v_avoid, w_avoid
        
        return None, None
    
    def calculate_path_alignment(self, trajectory, path):
        """计算轨迹与全局路径的对齐程度"""
        if len(path) == 0:
            return 0
        
        total_alignment = 0
        count = 0
        
        for point in trajectory:
            # 找到路径上最近的点
            distances = [np.linalg.norm(point - np.array(p)) for p in path]
            min_dist = min(distances)
            # 距离越小，对齐度越高
            alignment = max(0, 1.0 - min_dist / 1.5)
            total_alignment += alignment
            count += 1
        
        return total_alignment / count if count > 0 else 0
    
    def dwa_safety_verification(self, state, v_pp, w_pp, path, obstacles):
        """
        用DWA验证Pure Pursuit命令的安全性
        如果不安全，生成安全的替代命令
        """
        if obstacles is None or len(obstacles) == 0:
            return True, v_pp, w_pp
        
        self.dwa_calls += 1
        start_time = time.time()
        
        # 首先检查Pure Pursuit轨迹是否安全
        pp_trajectory = self.predict_trajectory(state, v_pp, w_pp)
        pp_safe = True
        
        for point in pp_trajectory:
            for obs in obstacles:
                dist = np.linalg.norm(point - obs[:2])
                if dist < obs[2] + self.robot_radius:
                    pp_safe = False
                    break
            if not pp_safe:
                break
        
        # 如果Pure Pursuit安全，但风险较高，也考虑替代方案
        if pp_safe:
            # 计算当前风险
            safety_score, _ = self.check_collision_risk(state, obstacles)
            if safety_score > self.obstacle_cost_threshold:
                return True, v_pp, w_pp
        
        # 如果不安全或风险高，用DWA搜索安全命令
        # 速度采样范围 - 扩大搜索范围
        v_min = max(0.1, v_pp - 1.0)
        v_max = min(self.max_speed, v_pp + 1.0)
        v_samples = np.linspace(v_min, v_max, self.v_samples)
        
        w_min = max(-self.max_yawrate, w_pp - 2.0)
        w_max = min(self.max_yawrate, w_pp + 2.0)
        w_samples = np.linspace(w_min, w_max, self.w_samples)
        
        best_v, best_w = v_pp, w_pp
        best_score = -float('inf')
        found_safe = False
        
        # 获取目标方向
        current_pos = state[:2]
        distances = [np.linalg.norm(np.array(p) - current_pos) for p in path]
        closest_idx = np.argmin(distances)
        target_idx = min(closest_idx + 3, len(path) - 1)
        target_point = np.array(path[target_idx])
        target_dir = target_point - current_pos
        if np.linalg.norm(target_dir) > 0:
            target_dir = target_dir / np.linalg.norm(target_dir)
        
        # 遍历所有速度组合
        for v in v_samples:
            for w in w_samples:
                # 预测未来轨迹
                trajectory = self.predict_trajectory(state, v, w)
                
                # 安全检查 - 检查整个轨迹
                safe = True
                min_obs_dist = float('inf')
                
                for point in trajectory:
                    for obs in obstacles:
                        dist = np.linalg.norm(point - obs[:2])
                        min_obs_dist = min(min_obs_dist, dist)
                        if dist < obs[2] + self.robot_radius:
                            safe = False
                            break
                    if not safe:
                        break
                
                if not safe:
                    continue
                
                found_safe = True
                
                # 计算方向得分
                last_point = trajectory[-1]
                move_dir = last_point - current_pos
                if np.linalg.norm(move_dir) > 0:
                    move_dir = move_dir / np.linalg.norm(move_dir)
                    dir_score = max(0, np.dot(move_dir, target_dir))
                else:
                    dir_score = 0
                
                # 安全得分 - 距离障碍物越远越好
                safety_score = min(1.0, min_obs_dist / (self.robot_radius + self.safety_margin + 0.8))
                
                # 路径对齐得分
                alignment_score = self.calculate_path_alignment(trajectory, path)
                
                # 速度偏好 - 尽可能快
                speed_score = v / self.max_speed
                
                # 平滑性惩罚
                smoothness_penalty = 0.3 * abs(v - v_pp) + 0.2 * abs(w - w_pp)
                
                # 综合评分
                score = (self.collision_weight * safety_score + 
                        self.path_weight * dir_score + 
                        0.5 * alignment_score +
                        self.speed_weight * speed_score -
                        smoothness_penalty)
                
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w
        
        self.dwa_time += (time.time() - start_time)
        
        # 如果没有找到安全命令，使用紧急避障
        if not found_safe:
            v_avoid, w_avoid = self.compute_avoidance_command(state, obstacles)
            if v_avoid is not None:
                self.override_count += 1
                return False, v_avoid, w_avoid
            else:
                # 紧急停车
                self.override_count += 1
                return False, 0.0, 0.0
        
        # 如果找到的命令与PP不同，记录override
        if abs(best_v - v_pp) > 0.1 or abs(best_w - w_pp) > 0.2:
            self.override_count += 1
            return False, best_v, best_w
        else:
            return True, v_pp, w_pp
    
    def plan(self, state, global_path, obstacles):
        """
        主规划函数
        """
        self.step_count += 1
        
        # 1. 检查碰撞风险
        safety, threat_obs = self.check_collision_risk(state, obstacles)
        
        # 2. 如果非常危险，进入紧急模式
        if safety < 0.15:  # 非常危险
            # 使用避障命令
            v_avoid, w_avoid = self.compute_avoidance_command(state, obstacles)
            if v_avoid is not None:
                return v_avoid, w_avoid, False, 0
            
            # 否则紧急停车
            return 0.0, 0.0, False, 0
        
        # 3. 计算Pure Pursuit命令
        pp_start = time.time()
        v_pp, w_pp, yaw_error, target = self.pure_pursuit_control(state, global_path)
        self.pp_time += (time.time() - pp_start)
        
        # 4. 如果中等风险，减速
        if safety < 0.6:
            v_pp *= safety
        
        # 5. 低频DWA验证 - 提高触发频率
        # 每10步验证一次，或者在风险较高时验证
        dwa_trigger = (self.step_count % self.dwa_frequency == 0) or (safety < 0.8)
        
        if dwa_trigger and obstacles is not None and len(obstacles) > 0:
            safe, v_final, w_final = self.dwa_safety_verification(
                state, v_pp, w_pp, global_path, obstacles
            )
            
            # 如果DWA override了，打印信息
            if not safe:
                print(f"  [DWA] Override at step {self.step_count}: PP({v_pp:.2f},{w_pp:.2f}) -> DWA({v_final:.2f},{w_final:.2f})")
            
            return v_final, w_final, safe, yaw_error
        
        return v_pp, w_pp, True, yaw_error
    
    def get_stats(self):
        """获取性能统计"""
        return {
            'pp_time': self.pp_time,
            'dwa_time': self.dwa_time,
            'dwa_calls': self.dwa_calls,
            'override_count': self.override_count,
            'avg_pp_time': self.pp_time / max(1, self.step_count) * 1000,
            'avg_dwa_time': self.dwa_time / max(1, self.dwa_calls) * 1000,
        }