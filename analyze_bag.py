"""
Phase II bag analysis — works without a ROS2 install (uses mcap + mcap_ros2).
Usage:
    python analyze_bag.py [t0] [tf]
Plots saved to plots/rosbag/
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

BAG_PATH  = Path(__file__).parent / "rosbag" / "rosbag.mcap"
NAMESPACE = "crazy_jirl_b3"
OUT_DIR   = Path(__file__).parent / "plots" / "rosbag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Circle track waypoints from Phase II spec [x, y, z, roll, pitch, yaw]
WAYPOINTS = np.array([
    [ 0.0, 3.0, 0.75, 0.0, 0.0,  0.00],
    [-1.5, 4.5, 0.75, 0.0, 0.0, -1.57],
    [ 0.0, 6.0, 1.75, 0.0, 0.0,  3.14],
    [ 1.5, 4.5, 0.75, 0.0, 0.0,  1.57],
])

def set_axes_equal(ax):
    lims = [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]
    ranges = [abs(l[1]-l[0]) or 1.0 for l in lims]
    max_r = max(ranges) / 2
    mids  = [(l[0]+l[1])/2 for l in lims]
    ax.set_xlim(mids[0]-max_r, mids[0]+max_r)
    ax.set_ylim(mids[1]-max_r, mids[1]+max_r)
    ax.set_zlim(mids[2]-max_r, mids[2]+max_r)

def gate_quads(waypoints, half=0.5):
    local = np.array([[0, half, half], [0, -half, half],
                      [0, -half, -half], [0, half, -half]])
    rots = R.from_euler('xyz', waypoints[:, 3:]).as_matrix()
    return np.einsum('ij,nkj->nik', local, rots) + waypoints[:, :3, None].transpose(0,2,1)

# ── read ──────────────────────────────────────────────────────────────────────

def read_bag(bag_path, ns, t0=0, tf=float('inf')):
    timestamps, gt_pos, gt_quat, gt_lin_vel, gt_ang_vel = \
        [], {k:[] for k in 'xyz'}, {k:[] for k in 'xyzw'}, \
        {k:[] for k in 'xyz'}, {k:[] for k in 'xyz'}
    ts_cmd, thr_pwm, thr_N, rr, pr, yr = [], [], [], [], [], []
    ts_traj, traj_x = [], []
    ts_obs, dist_gate = [], []

    odom_t = f"{ns}/odom"
    cmd_t  = "ctbr_cmd"
    traj_t = f"{ns}/trajectory"
    obs_t  = f"{ns}/observations"

    first = None
    print(f"Reading {bag_path} …")
    with open(bag_path, 'rb') as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, msg in reader.iter_decoded_messages():
            topic = channel.topic.lstrip('/')
            ts = message.log_time * 1e-9
            if first is None: first = ts
            rel = ts - first
            if rel < t0 or rel > tf:
                continue

            if topic == odom_t:
                timestamps.append(rel)
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                gt_pos['x'].append(p.x); gt_pos['y'].append(p.y); gt_pos['z'].append(p.z)
                gt_quat['x'].append(o.x); gt_quat['y'].append(o.y)
                gt_quat['z'].append(o.z); gt_quat['w'].append(o.w)
                rot  = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix().T
                lv_b = rot @ [msg.twist.twist.linear.x,  msg.twist.twist.linear.y,  msg.twist.twist.linear.z]
                av_b = rot @ [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
                av_b *= 180 / np.pi
                for i, k in enumerate('xyz'):
                    gt_lin_vel[k].append(lv_b[i])
                    gt_ang_vel[k].append(av_b[i])

            elif topic == cmd_t:
                cf = getattr(msg, 'crazyflie_name', None)
                if cf is None or ns in str(cf):
                    ts_cmd.append(rel)
                    thr_pwm.append(msg.thrust_pwm); thr_N.append(msg.thrust_n)
                    rr.append(msg.roll_rate); pr.append(msg.pitch_rate); yr.append(msg.yaw_rate)

            elif topic == traj_t:
                ts_traj.append(rel)
                traj_x.append(list(msg.x))

            elif topic == obs_t:
                ts_obs.append(rel)
                corners = np.array(msg.corners_pos_b_curr).reshape(4, 3)
                dist_gate.append(float(np.linalg.norm(corners.mean(axis=0))))

    traj_x = np.array(traj_x) if traj_x else np.empty((0, 3))
    print(f"  odom:{len(timestamps)}  cmd:{len(ts_cmd)}  traj:{len(ts_traj)}  obs:{len(ts_obs)}")
    return (timestamps, gt_pos, gt_quat, gt_lin_vel, gt_ang_vel,
            ts_cmd, thr_pwm, thr_N, rr, pr, yr,
            ts_traj, traj_x,
            ts_obs, dist_gate)

# ── plots ─────────────────────────────────────────────────────────────────────

def plot_all(ns, out, timestamps, gt_pos, gt_quat, gt_lin_vel, gt_ang_vel,
             ts_cmd, thr_pwm, thr_N, rr, pr, yr,
             ts_traj, traj_x, ts_obs, dist_gate):

    euler = R.from_quat(np.column_stack([gt_quat[k] for k in 'xyzw'])).as_euler('xyz', degrees=True)

    # gate pass detection
    gate_passes = [ts_obs[i] for i in range(1, len(dist_gate))
                   if abs(dist_gate[i-1] - dist_gate[i]) > 0.3]

    vm = np.sqrt(sum(np.array(gt_lin_vel[k])**2 for k in 'xyz'))
    print(f"\nDuration : {timestamps[-1]:.2f} s")
    print(f"Speed    : mean={vm.mean():.3f} m/s, max={vm.max():.3f} m/s")
    print(f"Gate passes detected: {len(gate_passes)} at t={[f'{t:.2f}s' for t in gate_passes]}")

    # 1. Ground truth overview
    fig, axs = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"Ground Truth — {ns}")
    axs[0].plot(timestamps, gt_pos['x'], label='x')
    axs[0].plot(timestamps, gt_pos['y'], label='y')
    axs[0].plot(timestamps, gt_pos['z'], label='z')
    axs[0].set_ylabel("Position [m]"); axs[0].legend(); axs[0].grid(True)
    axs[1].plot(timestamps, euler[:,0], label='Roll')
    axs[1].plot(timestamps, euler[:,1], label='Pitch')
    axs[1].plot(timestamps, euler[:,2], label='Yaw')
    axs[1].set_ylabel("Euler [deg]"); axs[1].legend(); axs[1].grid(True)
    axs[2].plot(timestamps, gt_lin_vel['x'], label='vx')
    axs[2].plot(timestamps, gt_lin_vel['y'], label='vy')
    axs[2].plot(timestamps, gt_lin_vel['z'], label='vz')
    axs[2].set_ylabel("Lin Vel [m/s]"); axs[2].legend(); axs[2].grid(True)
    axs[3].plot(timestamps, gt_ang_vel['x'], label='ωx')
    axs[3].plot(timestamps, gt_ang_vel['y'], label='ωy')
    axs[3].plot(timestamps, gt_ang_vel['z'], label='ωz')
    axs[3].set_ylabel("Ang Vel [deg/s]"); axs[3].set_ylim(-250, 250)
    axs[3].set_xlabel("Time [s]"); axs[3].legend(); axs[3].grid(True)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out / f"{ns}_ground_truth.png", dpi=150, bbox_inches='tight')
    plt.close(fig); print("Saved: ground_truth.png")

    # 2. Angular velocity actual vs commanded
    if ts_cmd:
        fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        fig.suptitle(f"Angular Velocity: Actual vs Commanded — {ns}")
        for i, (label, actual, desired) in enumerate([
                ('Roll Rate [deg/s]',  gt_ang_vel['x'], rr),
                ('Pitch Rate [deg/s]', gt_ang_vel['y'], pr),
                ('Yaw Rate [deg/s]',   gt_ang_vel['z'], yr)]):
            axs[i].plot(timestamps, actual, label='Actual')
            axs[i].plot(ts_cmd, desired, '--', label='Commanded')
            axs[i].set_ylabel(label); axs[i].set_ylim(-250, 250)
            axs[i].legend(); axs[i].grid(True)
        axs[-1].set_xlabel("Time [s]")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(out / f"{ns}_angular_velocity.png", dpi=150, bbox_inches='tight')
        plt.close(fig); print("Saved: angular_velocity.png")

    # 3. CTBR commands
    if ts_cmd:
        fig, axs = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        fig.suptitle(f"CTBR Commands — {ns}")
        axs[0].plot(ts_cmd, thr_pwm); axs[0].set_ylabel("Thrust PWM"); axs[0].grid(True)
        axs[1].plot(ts_cmd, thr_N);   axs[1].set_ylabel("Thrust [N]"); axs[1].grid(True)
        axs[2].plot(ts_cmd, rr, label='Roll')
        axs[2].plot(ts_cmd, pr, label='Pitch')
        axs[2].plot(ts_cmd, yr, label='Yaw')
        axs[2].set_ylabel("Body Rates [deg/s]"); axs[2].set_ylim(-250, 250)
        axs[2].set_xlabel("Time [s]"); axs[2].legend(); axs[2].grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(out / f"{ns}_ctbr_commands.png", dpi=150, bbox_inches='tight')
        plt.close(fig); print("Saved: ctbr_commands.png")

    # 4. Gate distance over time
    if ts_obs:
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(ts_obs, dist_gate, label='Distance to next gate')
        for tp in gate_passes:
            ax.axvline(tp, color='red', linestyle='--', alpha=0.6)
        if gate_passes:
            from matplotlib.lines import Line2D
            ax.legend([ax.lines[0], Line2D([0],[0], color='red', linestyle='--')],
                      ['Distance to next gate', f'Gate pass ({len(gate_passes)} total)'])
        ax.set_ylabel("Distance [m]"); ax.set_xlabel("Time [s]")
        ax.set_title(f"Gate Distance — {ns}"); ax.grid(True)
        fig.tight_layout()
        fig.savefig(out / f"{ns}_gate_distance.png", dpi=150, bbox_inches='tight')
        plt.close(fig); print("Saved: gate_distance.png")

    # 5. 3D trajectory
    quads = gate_quads(WAYPOINTS)
    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.plot(gt_pos['x'], gt_pos['y'], gt_pos['z'], 'b-', lw=1.5, label='Actual')
    if traj_x.shape[0] > 0:
        ax3.plot(traj_x[:,0], traj_x[:,1], traj_x[:,2], 'r--', lw=1, alpha=0.6, label='Desired')
    if gate_passes:
        ta = np.array(timestamps)
        xa, ya, za = np.array(gt_pos['x']), np.array(gt_pos['y']), np.array(gt_pos['z'])
        for i, tp in enumerate(gate_passes):
            idx = np.argmin(np.abs(ta - tp))
            ax3.scatter(xa[idx], ya[idx], za[idx], c='red', s=60,
                        label='Gate pass' if i == 0 else None)
    for verts in quads:
        ax3.add_collection3d(Poly3DCollection([verts], color='cyan', alpha=0.3, edgecolor='k'))
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
    ax3.set_title(f"3D Trajectory — {ns}"); ax3.legend()
    set_axes_equal(ax3)
    fig.savefig(out / f"{ns}_trajectory_3d.png", dpi=150, bbox_inches='tight')
    plt.close(fig); print("Saved: trajectory_3d.png")

    # 6. Top-down XY with gate markers
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt_pos['x'], gt_pos['y'], 'b-', lw=1.5, label='Actual')
    if traj_x.shape[0] > 0:
        ax.plot(traj_x[:,0], traj_x[:,1], 'r--', lw=1, alpha=0.6, label='Desired')
    for i, wp in enumerate(WAYPOINTS):
        ax.plot(wp[0], wp[1], 'c^', ms=12, zorder=5)
        ax.annotate(f"WP{i+1}", (wp[0], wp[1]), textcoords='offset points',
                    xytext=(5, 5), fontsize=9, color='teal')
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Top-Down Trajectory — {ns}")
    ax.set_aspect('equal'); ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(out / f"{ns}_topdown.png", dpi=150, bbox_inches='tight')
    plt.close(fig); print("Saved: topdown.png")

    print(f"\nAll plots → {out}/")

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0
    tf = float(sys.argv[2]) if len(sys.argv) > 2 else float('inf')
    data = read_bag(BAG_PATH, NAMESPACE, t0, tf)
    plot_all(NAMESPACE, OUT_DIR, *data)
