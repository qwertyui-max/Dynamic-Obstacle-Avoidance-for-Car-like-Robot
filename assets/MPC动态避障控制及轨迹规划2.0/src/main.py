import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from config.params import PURE_PURSUIT_CONFIG, DWA_SAFETY_CONFIG, MPC_CONFIG, SIMULATION_PARAMS
from planning.global_planner import AStarPlanner
from planning.hybrid_pp_dwa import HybridPP_DWA
from control.motion_controller import mpc_controller
from control.pid_controller import PIDController
from simulator.mujoco_simulator import get_car_state, check_collision, extract_obstacles
from utils.visualization import Visualizer
from utils.geometry import normalize_angle


def main():
    # 加载参数
    config = {**PURE_PURSUIT_CONFIG, **MPC_CONFIG}
    sim_params = SIMULATION_PARAMS
    robot_radius = config.get('robot_radius', 0.5)

    print("=" * 60)
    print("Pure Pursuit + DWA Hybrid Motion Planning System")
    print("=" * 60)

    # 初始化仿真以提取正确的障碍物
    print("\n[1/6] Extracting obstacles from MuJoCo model...")
    temp_model = mujoco.MjModel.from_xml_path(r"ddr.xml")
    obstacles = extract_obstacles(temp_model)
    print(f"Extracted {len(obstacles)} obstacles")
    for i, obs in enumerate(obstacles[:5]):
        print(f"  Obstacle {i}: pos=({obs[0]:.2f}, {obs[1]:.2f}), radius={obs[2]:.3f}")

    # A* 路径规划
    print("\n[2/6] Starting A* path planning...")
    astar_planner = AStarPlanner(grid_size=0.25, robot_radius=robot_radius)
    global_path = astar_planner.plan_path(
        sim_params['start_pos'],
        sim_params['goal_pos'],
        obstacles
    )

    if global_path is None:
        print("No valid path found!")
        return

    print(f"Path found with {len(global_path)} waypoints")
    for i, wp in enumerate(global_path):
        print(f"  Waypoint {i}: ({wp[0]:.2f}, {wp[1]:.2f})")

    # 初始化混合规划器
    print("\n[3/6] Initializing Pure Pursuit + DWA hybrid planner...")
    hybrid_planner = HybridPP_DWA(PURE_PURSUIT_CONFIG, DWA_SAFETY_CONFIG)

    # 初始化仿真
    print("[4/6] Initializing MuJoCo simulation...")
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
    print("[5/6] Setting up visualization...")
    visualizer = Visualizer(
        obstacles,
        sim_params['start_pos'],
        sim_params['goal_pos'],
        global_path
    )

    # 数据收集
    trajectory = []
    controls = []
    timestamps = []
    pp_commands = []
    mpc_commands = []
    override_events = []
    distance_history = []

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

    # 性能监控
    stuck_counter = 0
    prev_distance = float('inf')
    last_progress_step = 0
    no_progress_count = 0

    print("\n[6/6] Starting main simulation loop...")
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
            distance_history.append(dist_to_goal)

            # 检查碰撞
            if check_collision(model, data):
                collision_count += 1
                print(f"⚠️ Collision detected at step {step}! (Total: {collision_count})")
                # 碰撞后稍微后退
                data.ctrl[0] = -0.3
                data.ctrl[1] = 0.0
                mujoco.mj_step(model, data)
                continue

            # 状态机
            if dist_to_goal < sim_params['stopping_threshold']:
                state_machine = STOPPING
            elif emergency_stop:
                state_machine = EMERGENCY
            else:
                state_machine = NORMAL

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

                if abs(state[3]) < sim_params['stopping_speed_threshold'] and dist_to_goal < 0.5:
                    print(f"✅ Successfully stopped at goal in {step} steps!")
                    break

            elif state_machine == EMERGENCY:
                # 紧急停车
                data.ctrl[0] = -1.0
                data.ctrl[1] = 0.0
                mujoco.mj_step(model, data)
                controls.append([-1.0, 0.0])

                if abs(state[3]) < 0.01:
                    emergency_stop = False
                    print("Emergency stop cleared")

            else:  # 正常控制
                # 运行混合规划器
                if step % sim_params['high_level_freq'] == 0:
                    # Pure Pursuit + DWA 混合规划
                    v_pp, w_pp, safe, yaw_error = hybrid_planner.plan(
                        state, global_path, obstacles
                    )

                    pp_commands.append((v_pp, w_pp))

                    if not safe:
                        override_events.append(step)
                        print(f"  ↪ DWA override at step {step}, yaw_error={np.degrees(yaw_error):.1f}°")

                    # 生成参考轨迹用于MPC
                    ref_traj = np.tile(state, (config['horizon'], 1))

                    # MPC优化
                    try:
                        desired_v, desired_w = mpc_controller(
                            state, ref_traj, v_pp, w_pp, config, obstacles
                        )
                    except Exception as e:
                        # 如果MPC失败，使用Pure Pursuit的命令
                        desired_v, desired_w = v_pp, w_pp

                    mpc_commands.append((desired_v, desired_w))

                    # 滤波
                    desired_v = filter_alpha * prev_desired_v + (1 - filter_alpha) * desired_v
                    desired_w = filter_alpha * prev_desired_w + (1 - filter_alpha) * desired_w

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

            # 进度指示 - 每50步输出一次
            if step % 50 == 0:
                stats = hybrid_planner.get_stats()
                progress = distance_history[0] - dist_to_goal if distance_history else 0

                # 计算最近50步的进展
                if len(distance_history) > 50:
                    recent_progress = distance_history[-50] - dist_to_goal
                    if recent_progress < 0.1 and state[3] > 0.1:
                        # 进展缓慢但仍在移动，只是提醒
                        print(f"Step {step:4d} | Pos: ({state[0]:.1f}, {state[1]:.1f}) | "
                              f"Dist: {dist_to_goal:.2f}m | Speed: {state[3]:.2f}m/s | "
                              f"Progress: {progress:.2f}m | 进展缓慢但正常移动")
                    elif recent_progress < 0.05 and state[3] < 0.1:
                        # 确实卡住，尝试调整
                        no_progress_count += 1
                        print(f"Step {step:4d} | Pos: ({state[0]:.1f}, {state[1]:.1f}) | "
                              f"Dist: {dist_to_goal:.2f}m | Speed: {state[3]:.2f}m/s | "
                              f"Progress: {progress:.2f}m | ⚠️ 可能卡住，尝试调整")
                        # 微调参数
                        hybrid_planner.turn_gain = min(2.5, hybrid_planner.turn_gain * 1.1)
                    else:
                        # 正常进展
                        print(f"Step {step:4d} | Pos: ({state[0]:.1f}, {state[1]:.1f}) | "
                              f"Dist: {dist_to_goal:.2f}m | Speed: {state[3]:.2f}m/s | "
                              f"Progress: {progress:.2f}m | DWA overrides: {stats['override_count']}")
                        no_progress_count = 0
                else:
                    print(f"Step {step:4d} | Pos: ({state[0]:.1f}, {state[1]:.1f}) | "
                          f"Dist: {dist_to_goal:.2f}m | Speed: {state[3]:.2f}m/s | "
                          f"Progress: {progress:.2f}m | DWA overrides: {stats['override_count']}")

            # 检查是否应该终止（到达终点或长时间无进展）
            if dist_to_goal < sim_params['goal_threshold']:
                print(f"✅ Reached goal at step {step}!")
                break

            # 只有完全停滞时才终止
            if no_progress_count > 5 and state[3] < 0.05:  # 连续250步几乎没动且速度极低
                print(f"⚠️ Robot appears stuck, terminating simulation")
                break

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
    total_progress = distance_history[0] - distance_history[-1] if distance_history else 0

    print(f"\n📊 Performance Statistics:")
    print(f"  • Total steps: {step}")
    print(f"  • Collisions: {collision_count}")
    print(f"  • Final distance to goal: {distance_history[-1]:.2f}m")
    print(f"  • Total progress: {total_progress:.2f}m")
    print(f"  • DWA safety calls: {stats['dwa_calls']}")
    print(f"  • DWA override events: {stats['override_count']}")
    print(f"\n⏱️  Timing:")
    print(f"  • Avg Pure Pursuit time: {stats['avg_pp_time']:.3f} ms")
    print(f"  • Avg DWA time: {stats['avg_dwa_time']:.3f} ms")
    print(f"  • Total simulation time: {time.time() - start_time:.2f} s")

    # 最终可视化
    if len(trajectory) > 0:
        trajectory = np.array(trajectory)
        controls = np.array(controls) if controls else np.array([])
        pp_commands = np.array(pp_commands) if pp_commands else np.array([])
        mpc_commands = np.array(mpc_commands) if mpc_commands else np.array([])

        hl_times = np.array([i * sim_params['high_level_freq'] * model.opt.timestep
                             for i in range(len(pp_commands))]) if len(pp_commands) > 0 else np.array([])

        visualizer.final_plot(trajectory, controls, timestamps, pp_commands, mpc_commands, hl_times)

        # 路径效率分析
        if len(trajectory) > 1:
            path_length = np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1))
            straight_distance = np.linalg.norm(trajectory[0, :2] - trajectory[-1, :2])
            efficiency = straight_distance / path_length if path_length > 0 else 0

            print(f"\n📈 Path Efficiency:")
            print(f"  • Path length: {path_length:.2f} m")
            print(f"  • Straight distance: {straight_distance:.2f} m")
            print(f"  • Efficiency: {efficiency:.2%}")


if __name__ == "__main__":
    main()