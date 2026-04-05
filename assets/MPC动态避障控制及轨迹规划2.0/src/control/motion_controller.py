import numpy as np
from scipy.optimize import minimize
from utils.geometry import normalize_angle
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def car_kinematic_model(state, v_cmd, w_cmd, dt):
    x, y, yaw, v_curr, w_curr = state
    tau_v = 0.2
    tau_w = 0.1
    
    # Ensure dt is positive and reasonable
    dt = max(0.01, min(0.1, dt))
    
    # Limit command rates for stability
    max_v_change = 2.0 * dt
    max_w_change = 5.0 * dt
    
    v_cmd = np.clip(v_cmd, v_curr - max_v_change, v_curr + max_v_change)
    w_cmd = np.clip(w_cmd, w_curr - max_w_change, w_curr + max_w_change)
    
    v_next = v_curr + (v_cmd - v_curr) * min(1.0, dt / tau_v)
    w_next = w_curr + (w_cmd - w_curr) * min(1.0, dt / tau_w)
    
    x_new = x + v_next * np.cos(yaw) * dt
    y_new = y + v_next * np.sin(yaw) * dt
    yaw_new = yaw + w_next * dt
    
    return np.array([x_new, y_new, normalize_angle(yaw_new), v_next, w_next])

def mpc_controller(current_state, ref_trajectory, v_lspp, w_lspp, config, obstacles):
    horizon = config['horizon']
    dt_mpc = config['dt_mpc']
    max_speed = config['max_speed']
    max_yawrate = config['max_yawrate']
    weights = config['weights_mpc']
    robot_radius = config.get('robot_radius', 0.8)
    
    # Safety checks
    if np.any(np.isnan(current_state)) or np.any(np.isinf(current_state)):
        print("Warning: Invalid current state detected")
        return v_lspp, w_lspp
    
    # Prepare reference trajectory with safety checks
    if ref_trajectory is None or len(ref_trajectory) == 0:
        ref_traj = np.tile(current_state, (horizon, 1))
    else:
        if len(ref_trajectory) < horizon:
            last_state = ref_trajectory[-1] if len(ref_trajectory) > 0 else current_state
            padding = np.tile(last_state, (horizon - len(ref_trajectory), 1))
            ref_traj = np.vstack([ref_trajectory, padding])
        else:
            ref_traj = ref_trajectory[:horizon]
    
    def cost_function(u):
        try:
            v_cmds = u[:horizon]
            w_cmds = u[horizon:]
            cost = 0.0
            state = current_state.copy()
            
            for t in range(horizon):
                # Clip commands for stability
                v_cmd = np.clip(v_cmds[t], -max_speed, max_speed)
                w_cmd = np.clip(w_cmds[t], -max_yawrate, max_yawrate)
                
                state = car_kinematic_model(state, v_cmd, w_cmd, dt_mpc)
                
                # Check for invalid state
                if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                    return 1e10
                
                ref_state = ref_traj[t]
                
                # Position error cost
                pos_error = np.linalg.norm(state[:2] - ref_state[:2])
                cost += weights['position'] * pos_error
                
                # Heading error cost (minimize absolute difference)
                heading_error = abs(normalize_angle(state[2] - ref_state[2]))
                cost += weights['heading'] * heading_error
                
                # Obstacle avoidance cost
                obstacle_cost = calculate_obstacle_cost(state, obstacles, robot_radius)
                if obstacle_cost > 10:  # Cap obstacle cost
                    obstacle_cost = 10
                cost += weights['obstacle'] * obstacle_cost
                
                # Control smoothness costs
                if t > 0:
                    dv = abs(v_cmds[t] - v_cmds[t-1]) / max(dt_mpc, 0.01)
                    dw = abs(w_cmds[t] - w_cmds[t-1]) / max(dt_mpc, 0.01)
                    cost += weights['dv'] * min(dv, 10)  # Cap rate of change
                    cost += weights['dw'] * min(dw, 10)
                
                # Tracking of LSPP commands
                cost += weights['v'] * abs(v_cmds[t] - v_lspp)
                cost += weights['w'] * abs(w_cmds[t] - w_lspp)
                
                # Add penalty for extreme values
                cost += 0.01 * (v_cmd**2 + w_cmd**2)
            
            return cost
            
        except Exception as e:
            print(f"Cost function error: {e}")
            return 1e10
    
    # Set bounds with safety margins
    v_bounds = [(max(-max_speed, -2.0), min(max_speed, 2.0))] * horizon
    w_bounds = [(-max_yawrate, max_yawrate)] * horizon
    bounds = v_bounds + w_bounds
    
    # Initial guess with smoothing
    v_init = np.clip(v_lspp, -max_speed, max_speed)
    w_init = np.clip(w_lspp, -max_yawrate, max_yawrate)
    u0 = np.concatenate([np.ones(horizon) * v_init, 
                         np.ones(horizon) * w_init])
    
    # Add small perturbation to avoid local minima
    u0 += np.random.normal(0, 0.01, size=len(u0))
    
    # Clip to bounds
    u0_clipped = np.clip(u0, 
                         [b[0] for b in bounds], 
                         [b[1] for b in bounds])
    
    # Run optimization with robust settings
    try:
        res = minimize(cost_function, u0_clipped, method='SLSQP', bounds=bounds, 
                       options={'maxiter': 50, 'ftol': 1e-2, 'eps': 1e-3})
        
        if res.success and not np.any(np.isnan(res.x)):
            v_opt = np.clip(res.x[0], -max_speed, max_speed)
            w_opt = np.clip(res.x[horizon], -max_yawrate, max_yawrate)
        else:
            # Fallback to LSPP commands
            v_opt, w_opt = v_lspp, w_lspp
            
    except Exception as e:
        print(f"MPC optimization failed: {e}, using LSPP commands")
        v_opt, w_opt = v_lspp, w_lspp
    
    return float(v_opt), float(w_opt)

def calculate_obstacle_cost(state, obstacles, robot_radius):
    """Simplified obstacle cost calculation for a single state"""
    if obstacles.size == 0:
        return 0.0
    
    pos = state[:2]
    min_dist = float('inf')
    
    for obstacle in obstacles:
        dist = np.linalg.norm(pos - obstacle[:2])
        if dist < min_dist:
            min_dist = dist
    
    safety_margin = robot_radius + 0.2
    
    if min_dist < safety_margin:
        return 10.0  # Cap instead of inf
    elif min_dist < safety_margin + 0.8:
        return 1.0 / max(min_dist - safety_margin, 0.1)
    else:
        return 0.0