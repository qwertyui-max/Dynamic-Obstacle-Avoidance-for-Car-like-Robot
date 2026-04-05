import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from config.params import LSPP_CONFIG, DWA_SAFETY_CONFIG, MPC_CONFIG, SIMULATION_PARAMS
from planning.global_planner import AStarPlanner
from planning.hybrid_lspp_dwa import HybridLSPP_DWA
from control.motion_controller import mpc_controller
from control.pid_controller import PIDController
from simulator.mujoco_simulator import get_car_state, check_collision
from utils.visualization import Visualizer
from utils.geometry import normalize_angle

def main():
    # 加载参数
    config = {**LSPP_CONFIG, **MPC_CONFIG}
    sim_params = SIMULATION_PARAMS
    robot_radius = config.get('robot_radius', 0.8)
    
    print("=" * 60)
    print("LSPP + DWA Hybrid Motion Planning System")
    print("=" * 60)
    
    # A* 路径规划
    print("\n[1/5] Starting A* path planning...")
    astar_planner = AStarPlanner(grid_size=0.25, robot_radius=robot_radius)
    global_path = astar_planner.plan_path(
        sim_params['start_pos'], 
        sim_params['goal_pos'], 
        sim_params['obstacles']
    )
    
    if global_path is None:
        print("No valid path found!")
        return
    
    print(f"Path found with {len(global_path)} waypoints")
    
    # 初始化混合规划器
    print("\n[2/5] Initializing LSPP+DWA hybrid planner...")
    hybrid_planner = HybridLSPP_DWA(LSPP_CONFIG, DWA_SAFETY_CONFIG)
    
    # 初始化仿真
    print("[3/5] Initializing MuJoCo simulation...")
    model = mujoco.MjModel.from_xml_path(r"ddr.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = 0.02
    
    # 设置初始状态
    data.qpos[0:2] = sim_params['start_pos']
    quat = R.from_euler('z', sim_params['start_yaw']).as_quat()
    data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
    data.qvel[:] = 0
    
    # 初始化PID控制器
    v_pid = PIDController(kp=0.8, ki=0.1, kd=0.02)
    w_pid = PIDController(kp=0.5, ki=0.2, kd=0.01)
    
    # 初始化可视化
    print("[4/5] Setting up visualization...")
    visualizer = Visualizer(
        sim_params['obstacles'], 
        sim_params['start_pos'], 
        sim_params['goal_pos'],
        global_path
    )
    
    # 数据收集
    trajectory = []
    controls = []
    timestamps = []
    lspp_commands = []
    mpc_commands = []
    safety_scores = []
    lyapunov_values = []
    override_events = []
    
    desired_v = 0.0
    desired_w = 0.0
    prev_desired_v = 0.0
    prev_desired_w = 0.0
    filter_alpha = 0.7
    
    start_time = time.time()
    
    # 初始化viewer
    try:
        from mujoco import viewer
        viewer_available = True
        viewer = viewer.launch_passive(model, data)
        viewer.sync()
        print("Viewer launched successfully")
    except Exception as e:
        print(f"Could not create viewer: {e}")
        viewer_available = False
    
    # 状态机
    STOPPING = 0
    NORMAL = 1
    EMERGENCY = 2
    state_machine = NORMAL
    emergency_stop = False
    collision_count = 0
    
    print("\n[5/5] Starting main simulation loop...")
    print("=" * 60)
    
    # 主循环
    for step in range(sim_params['max_steps']):
        try:
            state = get_car_state(data)
            trajectory.append(state)
            timestamps.append(step * model.opt.timestep)
            
            # 检查无效状态
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print(f"Invalid state detected at step {step}")
                break
            
            # 计算到终点的距离
            dist_to_goal = np.linalg.norm(state[:2] - sim_params['goal_pos'])
            
            # 检查碰撞
            if check_collision(model, data):
                collision_count += 1
                if not emergency_stop:
                    print(f"⚠️ Emergency stop at step {step} - Collision detected!")
                    emergency_stop = True
                state_machine = EMERGENCY
            elif dist_to_goal < sim_params['stopping_threshold']:
                state_machine = STOPPING
            else:
                state_machine = NORMAL
                emergency_stop = False
            
            if state_machine == STOPPING:
                # 停车控制器
                desired_v = 0
                desired_w = 0
                u_forward = -0.5 * state[3]
                u_turn = -0.3 * state[4]
                
                data.ctrl[0] = u_forward
                data.ctrl[1] = u_turn
                controls.append([u_forward, u_turn])
                
                mujoco.mj_step(model, data)
                
                if abs(state[3]) < sim_params['stopping_speed_threshold'] and dist_to_goal < 0.3:
                    print(f"✅ Successfully stopped at goal in {step} steps!")
                    break
                    
            elif state_machine == EMERGENCY:
                # 紧急停车
                data.ctrl[0] = -1.0
                data.ctrl[1] = 0.0
                mujoco.mj_step(model, data)
                controls.append([-1.0, 0.0])
                
                if abs(state[3]) < 0.01:
                    print("Emergency stop complete")
                    break
                
            else:  # 正常控制
                # 运行混合规划器
                if step % sim_params['high_level_freq'] == 0:
                    # LSPP + DWA 混合规划
                    v_lspp, w_lspp, safe, lyapunov = hybrid_planner.plan(
                        state, global_path, sim_params['obstacles']
                    )
                    
                    lspp_commands.append((v_lspp, w_lspp))
                    lyapunov_values.append(lyapunov)
                    
                    if not safe:
                        override_events.append(step)
                    
                    # 生成参考轨迹用于MPC
                    ref_traj = np.tile(state, (config['horizon'], 1))
                    
                    # MPC优化
                    try:
                        desired_v, desired_w = mpc_controller(
                            state, ref_traj, v_lspp, w_lspp, config, sim_params['obstacles']
                        )
                    except Exception as e:
                        print(f"MPC error at step {step}, using LSPP")
                        desired_v, desired_w = v_lspp, w_lspp
                    
                    mpc_commands.append((desired_v, desired_w))
                    
                    # 滤波
                    desired_v = filter_alpha * prev_desired_v + (1-filter_alpha) * desired_v
                    desired_w = filter_alpha * prev_desired_w + (1-filter_alpha) * desired_w
                    
                    # 速率限制
                    max_v_change = 0.5 * model.opt.timestep * sim_params['high_level_freq']
                    max_w_change = 2.0 * model.opt.timestep * sim_params['high_level_freq']
                    
                    desired_v = np.clip(desired_v, 
                                       prev_desired_v - max_v_change,
                                       prev_desired_v + max_v_change)
                    desired_w = np.clip(desired_w,
                                       prev_desired_w - max_w_change,
                                       prev_desired_w + max_w_change)
                    
                    prev_desired_v = desired_v
                    prev_desired_w = desired_w
                
                # PID控制
                dt_sim = model.opt.timestep
                v_error = desired_v - state[3]
                w_error = desired_w - state[4]
                
                u_forward = v_pid.compute(v_error, dt_sim)
                u_turn = w_pid.compute(w_error, dt_sim)
                
                u_forward = np.clip(u_forward, -0.8, 0.8)
                u_turn = np.clip(u_turn, -0.8, 0.8)
                
                data.ctrl[0] = u_forward
                data.ctrl[1] = u_turn
                controls.append([u_forward, u_turn])
                
                mujoco.mj_step(model, data)
            
            # 更新viewer
            if viewer_available and step % 5 == 0:
                viewer.sync()
                time.sleep(model.opt.timestep)
            
            # 更新可视化
            if step % 20 == 0 and len(trajectory) > 1:
                visualizer.update(trajectory)
            
            # 进度指示
            if step % 100 == 0:
                stats = hybrid_planner.get_stats()
                print(f"Step {step:4d} | Dist: {dist_to_goal:.2f}m | "
                      f"Speed: {state[3]:.2f}m/s | DWA calls: {stats['dwa_calls']} | "
                      f"Instability: {stats['instability_count']}")
                
        except Exception as e:
            print(f"Error in main loop at step {step}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # 清理
    if viewer_available:
        viewer.close()
    
    # 最终统计
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    stats = hybrid_planner.get_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"  • Total steps: {step}")
    print(f"  • Collisions: {collision_count}")
    print(f"  • DWA safety calls: {stats['dwa_calls']}")
    print(f"  • DWA override events: {len(override_events)}")
    print(f"  • Instability events: {stats['instability_count']}")
    print(f"\n⏱️  Timing:")
    print(f"  • Avg LSPP time: {stats['avg_lspp_time']*1000:.3f} ms")
    print(f"  • Avg DWA time: {stats['avg_dwa_time']*1000:.3f} ms")
    print(f"  • Total simulation time: {time.time() - start_time:.2f} s")
    
    # 最终可视化
    if len(trajectory) > 0:
        trajectory = np.array(trajectory)
        controls = np.array(controls) if controls else np.array([])
        lspp_commands = np.array(lspp_commands) if lspp_commands else np.array([])
        mpc_commands = np.array(mpc_commands) if mpc_commands else np.array([])
        
        hl_times = np.array([i * sim_params['high_level_freq'] * model.opt.timestep 
                            for i in range(len(lspp_commands))]) if len(lspp_commands) > 0 else np.array([])
        
        visualizer.final_plot(trajectory, controls, timestamps, lspp_commands, mpc_commands, hl_times)
        
        # 性能指标
        total_time = timestamps[-1] if timestamps else 0
        final_speed = trajectory[-1, 3] if len(trajectory) > 0 else 0
        path_length = np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)) if len(trajectory) > 1 else 0
        
        print(f"\n📈 Path Statistics:")
        print(f"  • Total time: {total_time:.2f} s")
        print(f"  • Path length: {path_length:.2f} m")
        print(f"  • Average speed: {path_length/total_time:.2f} m/s")
        print(f"  • Final speed: {final_speed:.2f} m/s")

if __name__ == "__main__":
    main()