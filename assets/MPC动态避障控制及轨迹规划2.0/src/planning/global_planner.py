import numpy as np
import heapq
import math

class AStarPlanner:
    def __init__(self, grid_size=0.15, robot_radius=0.4):  # 减小网格大小
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        
    def create_grid(self, obstacles, xlim=(-2, 20), ylim=(-2, 15)):  # 扩大范围
        """创建占用网格，考虑机器人半径和障碍物实际大小"""
        min_x, max_x = xlim
        min_y, max_y = ylim
        width = int((max_x - min_x) / self.grid_size) + 1
        height = int((max_y - min_y) / self.grid_size) + 1
        
        grid = np.zeros((width, height))
        
        # 膨胀障碍物（考虑机器人半径 + 安全距离）
        safety_distance = 0.2  # 减小安全距离，让路径更接近障碍物
        
        for obs in obstacles:
            x, y, r = obs
            # 膨胀后的半径 = 障碍物半径 + 机器人半径 + 安全距离
            inflated_r = r + self.robot_radius + safety_distance
            
            # 计算网格范围
            obs_min_x = int((x - inflated_r - min_x) / self.grid_size)
            obs_max_x = int((x + inflated_r - min_x) / self.grid_size + 1)
            obs_min_y = int((y - inflated_r - min_y) / self.grid_size)
            obs_max_y = int((y + inflated_r - min_y) / self.grid_size + 1)
            
            obs_min_x = max(0, min(width-1, obs_min_x))
            obs_max_x = max(0, min(width-1, obs_max_x))
            obs_min_y = max(0, min(height-1, obs_min_y))
            obs_max_y = max(0, min(height-1, obs_max_y))
            
            # 标记障碍物区域
            for i in range(obs_min_x, obs_max_x + 1):
                for j in range(obs_min_y, obs_max_y + 1):
                    grid[i, j] = 1
        
        return grid, (min_x, min_y)
    
    def world_to_grid(self, pos, grid_origin):
        x, y = pos
        min_x, min_y = grid_origin
        grid_x = int((x - min_x) / self.grid_size)
        grid_y = int((y - min_y) / self.grid_size)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos, grid_origin):
        grid_x, grid_y = grid_pos
        min_x, min_y = grid_origin
        x = min_x + grid_x * self.grid_size
        y = min_y + grid_y * self.grid_size
        return (x, y)
    
    def heuristic(self, a, b):
        """欧几里得距离启发式"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def a_star_search(self, start, goal, grid, grid_origin):
        start_grid = self.world_to_grid(start, grid_origin)
        goal_grid = self.world_to_grid(goal, grid_origin)
        
        width, height = grid.shape
        
        # 检查起点和终点是否在网格内
        if not (0 <= start_grid[0] < width and 0 <= start_grid[1] < height):
            print(f"Start out of bounds: {start_grid}")
            return None
        if not (0 <= goal_grid[0] < width and 0 <= goal_grid[1] < height):
            print(f"Goal out of bounds: {goal_grid}")
            return None
        
        # 如果起点或终点在障碍物中，找最近空闲点
        if grid[start_grid[0], start_grid[1]] == 1:
            print("Start in obstacle, finding nearest free cell...")
            start_grid = self.find_nearest_free(start_grid, grid)
            if start_grid is None:
                return None
            print(f"Found new start at {start_grid}")
        
        if grid[goal_grid[0], goal_grid[1]] == 1:
            print("Goal in obstacle, finding nearest free cell...")
            goal_grid = self.find_nearest_free(goal_grid, grid)
            if goal_grid is None:
                return None
            print(f"Found new goal at {goal_grid}")
        
        # 8方向移动，调整权重让路径更直
        neighbors = [
            (0, 1, 1.0),   # 上
            (1, 0, 1.0),   # 右
            (0, -1, 1.0),  # 下
            (-1, 0, 1.0),  # 左
            (1, 1, 1.4),   # 右上（对角线）
            (1, -1, 1.4),  # 右下
            (-1, 1, 1.4),  # 左上
            (-1, -1, 1.4)  # 左下
        ]
        
        close_set = set()
        came_from = {}
        gscore = {start_grid: 0}
        fscore = {start_grid: self.heuristic(start_grid, goal_grid)}
        oheap = []
        heapq.heappush(oheap, (fscore[start_grid], start_grid))
        
        # 记录方向，用于惩罚方向变化
        direction_from = {}
        direction_from[start_grid] = (0, 0)
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            
            if current == goal_grid:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                path = path[::-1]
                return [self.grid_to_world(p, grid_origin) for p in path]
            
            close_set.add(current)
            
            for i, j, move_cost in neighbors:
                neighbor = (current[0] + i, current[1] + j)
                
                if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                    if grid[neighbor[0], neighbor[1]] == 1:
                        continue
                    
                    # 基础移动成本
                    movement_cost = move_cost * self.grid_size * 8
                    
                    # 方向变化惩罚（鼓励直线）
                    turn_penalty = 0
                    if current in direction_from:
                        prev_dir = direction_from[current]
                        curr_dir = (i, j)
                        if prev_dir != (0, 0) and prev_dir != curr_dir:
                            turn_penalty = 2.0  # 增加方向变化惩罚
                    
                    # 距离目标越近，成本越低（鼓励向目标移动）
                    dist_to_goal = self.heuristic(neighbor, goal_grid)
                    goal_bonus = 0.3 * (1.0 - dist_to_goal / self.heuristic(start_grid, goal_grid))
                    
                    tentative_g_score = gscore[current] + movement_cost + turn_penalty - goal_bonus
                    
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                        continue
                    
                    if tentative_g_score < gscore.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        direction_from[neighbor] = (i, j)
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        return None
    
    def find_nearest_free(self, grid_pos, grid):
        """找到最近的空闲网格"""
        width, height = grid.shape
        search_radius = 1
        max_radius = 15
        
        while search_radius < max_radius:
            for i in range(max(0, grid_pos[0] - search_radius), 
                          min(width, grid_pos[0] + search_radius + 1)):
                for j in range(max(0, grid_pos[1] - search_radius), 
                              min(height, grid_pos[1] + search_radius + 1)):
                    if grid[i, j] == 0:
                        return (i, j)
            search_radius += 1
        
        return None
    
    def simplify_path(self, path):
        """简化路径，去除不必要的拐弯"""
        if len(path) < 3:
            return path
        
        def is_collinear(p1, p2, p3, tolerance=0.15):
            """检查三点是否共线"""
            v1 = np.array(p2) - np.array(p1)
            v2 = np.array(p3) - np.array(p2)
            
            if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
                return True
            
            # 计算叉积的模
            cross = abs(np.cross(v1, v2))
            # 归一化到距离
            cross = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return cross < tolerance
        
        simplified = [path[0]]
        
        for i in range(1, len(path) - 1):
            if not is_collinear(simplified[-1], path[i], path[i + 1]):
                simplified.append(path[i])
        
        simplified.append(path[-1])
        return simplified
    
    def interpolate_path(self, path, step_size=0.2):
        """插值路径，使路径更平滑"""
        if len(path) < 2:
            return path
        
        interpolated = [path[0]]
        
        for i in range(len(path) - 1):
            start = np.array(path[i])
            end = np.array(path[i + 1])
            dist = np.linalg.norm(end - start)
            
            if dist > step_size:
                num_points = int(dist / step_size)
                for j in range(1, num_points):
                    t = j / num_points
                    point = start + t * (end - start)
                    interpolated.append(tuple(point))
            
            interpolated.append(tuple(end))
        
        return interpolated
    
    def plan_path(self, start, goal, obstacles):
        """主路径规划函数"""
        print(f"Planning path from {start} to {goal}")
        
        # 创建网格
        grid, grid_origin = self.create_grid(obstacles)
        
        # 检查起点和终点
        start_grid = self.world_to_grid(start, grid_origin)
        goal_grid = self.world_to_grid(goal, grid_origin)
        
        print(f"Grid size: {grid.shape}")
        print(f"Start grid: {start_grid}")
        print(f"Goal grid: {goal_grid}")
        
        # 执行A*搜索
        path = self.a_star_search(start, goal, grid, grid_origin)
        
        if path:
            print(f"Raw path has {len(path)} waypoints")
            # 简化路径
            simplified = self.simplify_path(path)
            print(f"Simplified to {len(simplified)} waypoints")
            
            # 插值路径使更平滑
            interpolated = self.interpolate_path(simplified, step_size=0.15)
            print(f"Interpolated to {len(interpolated)} waypoints")
            
            return interpolated
        
        print("No path found!")
        return None