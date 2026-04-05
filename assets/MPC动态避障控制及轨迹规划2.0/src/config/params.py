import numpy as np

# Pure Pursuit 参数
PURE_PURSUIT_CONFIG = {
    'max_speed': 2.0,
    'min_speed': -0.3,
    'max_yawrate': 3.0,  # 降低角速度，使转向更平滑
    'max_accel': 0.5,
    'max_dyawrate': 2.0,
    'dt': 0.08,
    'predict_time': 1.5,
    'robot_radius': 0.4,  # 减小机器人半径，便于通过窄通道
    'lookahead_distance': 1.2,  # 增加前视距离
    'max_lookahead': 2.5,
    'min_lookahead': 0.6,
    'speed_gain': 0.7,
    'turn_gain': 1.5,
}

# DWA 安全验证参数 - 加强DWA的作用
DWA_SAFETY_CONFIG = {
    'v_samples': 12,  # 增加采样数
    'w_samples': 12,
    'predict_time': 1.5,
    'safety_margin': 0.4,  # 增加安全裕度
    'emergency_stop_dist': 0.8,
    'collision_weight': 15.0,  # 提高避障权重
    'path_weight': 2.0,  # 提高路径跟踪权重
    'speed_weight': 0.3,
    'obstacle_cost_threshold': 0.7,  # 触发DWA的阈值
}

MPC_CONFIG = {
    'horizon': 5,  # 增加预测时域
    'dt_mpc': 0.05,
    'max_speed': 2.0,
    'min_speed': -0.3,
    'max_yawrate': 3.0,
    'weights_mpc': {
        'position': 2.0,  # 提高位置权重
        'heading': 0.5,
        'v': 0.3,
        'w': 0.3,
        'dv': 1.5,  # 提高平滑性权重
        'dw': 1.2,
        'obstacle': 2.0  # 提高避障权重
    },
    'robot_radius': 0.4
}

SIMULATION_PARAMS = {
    'start_pos': [0, 0],
    'start_yaw': np.pi/2,
    'goal_pos': [9, 10],
    'max_steps': 3000,
    'goal_threshold': 0.4,
    'stopping_threshold': 1.0,
    'stopping_speed_threshold': 0.1,
    'high_level_freq': 2,
    'dwa_verify_freq': 10,  # 提高DWA验证频率（原来是30）
    'obstacles': np.array([
        [1.2, 10.8, 0.6], [16.8, 1.2, 0.7], [2.5, 2.5, 0.5], [15.5, 10.5, 0.6],
        [1.0, 6.0, 0.7], [17.0, 6.0, 0.6], [4.0, 11.0, 0.5], [14.0, 1.0, 0.7],
        [3.0, 8.5, 0.6], [15.0, 3.5, 0.5], [8.0, 1.5, 0.7], [10.0, 10.5, 0.6],
        [1.5, 3.5, 0.5], [16.5, 8.5, 0.7], [5.5, 5.0, 0.6], [12.5, 7.0, 0.5],
        [9.0, 8.0, 0.7], [9.0, 4.0, 0.6], [3.5, 6.5, 0.5], [14.5, 5.5, 0.7],
        [7.0, 9.0, 0.6], [11.0, 3.0, 0.5], [6.0, 2.0, 0.7], [12.0, 9.0, 0.6],
        [8.5, 6.0, 0.5]
    ])
}