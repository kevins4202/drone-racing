# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import subtract_frame_transforms, quat_from_euler_xyz, euler_xyz_from_quat, wrap_to_pi, matrix_from_quat

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, 'rew'):
            log_keys = ["pass", "crash", "cmd", "vel", "time"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in log_keys
            }

        # Distance buffer for progress reward (initialized to 0; clamped reward means no penalty on first step)
        self.env._last_distance_to_approach = torch.zeros(self.num_envs, device=self.device)

        # Initialize fixed parameters once (no domain randomization)
        # These parameters remain constant throughout the simulation
        # Aerodynamic drag coefficients
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        # PID controller gains for angular rate control
        # Roll and pitch use the same gains
        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        # Yaw has different gains
        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        # Motor time constants (same for all 4 motors)
        self.env._tau_m[:] = self.env._tau_m_value

        # Thrust to weight ratio
        self.env._thrust_to_weight[:] = self.env._twr_value

    def get_rewards(self) -> torch.Tensor:
        """Racing reward: pass (sparse), velocity-toward-gate (dense), time penalty, crash, cmd."""

        # Gate pass detection: x in gate frame crosses positive -> negative AND within aperture
        x_now = self.env._pose_drone_wrt_gate[:, 0]
        crossed = (self.env._prev_x_drone_wrt_gate > 0) & (x_now <= 0)
        in_aperture = (
            (torch.abs(self.env._pose_drone_wrt_gate[:, 1]) < 0.5) &
            (torch.abs(self.env._pose_drone_wrt_gate[:, 2]) < 0.5)
        )
        gate_passed = crossed & in_aperture

        # Update prev x for next step
        self.env._prev_x_drone_wrt_gate = x_now.clone()

        # Advance waypoint for envs that passed a gate
        ids = torch.where(gate_passed)[0]
        if len(ids) > 0:
            self.env._n_gates_passed[ids] += 1
            self.env._idx_wp[ids] = (self.env._idx_wp[ids] + 1) % self.env._waypoints.shape[0]
            self.env._prev_x_drone_wrt_gate[ids] = 1.0  # reset for new gate
            self.env._desired_pos_w[ids, :] = self.env._waypoints[self.env._idx_wp[ids], :3]

        # 1. Pass reward: sparse +1 per gate
        pass_reward = gate_passed.float()

        # 2. Crash penalty: contact force detection
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed * mask
        crash_reward = (self.env._crashed > 0).float()

        # 3. Cmd penalty: penalize large actions and action rate
        u = self.env._actions
        u_prev = self.env._previous_actions
        cmd_penalty = torch.norm(u, dim=1) + torch.norm(u - u_prev, dim=1) ** 2

        # 4. Velocity-toward-gate reward (replaces R_prog).
        # Rewards flying fast in the direction of the approach waypoint (1m in front of each gate
        # along its normal). Clamped to [0, 5] m/s: never penalizes detours (e.g. powerloop return)
        # and caps the signal so the drone can't "game" it by flying infinitely fast past the gate.
        drone_pos_w = self.env._robot.data.root_link_pos_w          # (N, 3)
        gate_pos = self.env._waypoints[self.env._idx_wp, :3]        # (N, 3)
        gate_normal = self.env._normal_vectors[self.env._idx_wp]     # (N, 3)
        approach_wp_w = gate_pos + gate_normal * 1.0                 # 1m on approach side
        d = approach_wp_w - drone_pos_w                              # (N, 3)
        d_hat = d / (torch.norm(d, dim=1, keepdim=True) + 1e-6)     # unit vector toward WP

        # Rotate body-frame velocity to world frame: v_w = R_WB @ v_b
        rot_mat = matrix_from_quat(self.env._robot.data.root_quat_w)     # (N, 3, 3)
        v_b = self.env._robot.data.root_com_lin_vel_b                    # (N, 3)
        v_world = torch.bmm(rot_mat, v_b.unsqueeze(-1)).squeeze(-1)      # (N, 3)

        vel_toward = (v_world * d_hat).sum(dim=1)                        # dot product (N,)
        vel_reward = torch.clamp(vel_toward, min=0.0, max=5.0)

        # 5. Time penalty: constant -1/step encourages fast lap completion.
        # Ensures hovering is never a stable local optimum.
        time_penalty = torch.ones(self.num_envs, device=self.device)

        if self.cfg.is_train:
            rewards = {
                "pass":  pass_reward  * self.env.rew['pass_reward_scale'],
                "crash": crash_reward * self.env.rew['crash_reward_scale'],
                "cmd":   cmd_penalty  * self.env.rew['cmd_reward_scale'],
                "vel":   vel_reward   * self.env.rew['vel_reward_scale'],
                "time":  time_penalty * self.env.rew['time_penalty_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
            reward = torch.where(self.env.reset_terminated,
                                 torch.ones_like(reward) * self.env.rew['death_cost'], reward)

            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """25-dim actor obs + 34-dim privileged critic obs (asymmetric actor-critic).

        Actor (25d): body vel, body rates, rotation matrix cols 1-2, current gate pos,
                     next gate pos, 2nd look-ahead gate pos, prev actions.
        Critic (34d): all actor obs + 9 normalized dynamics parameters.
        """
        drone_lin_vel_b = self.env._robot.data.root_com_lin_vel_b   # (N,3) body frame velocity
        drone_ang_vel_b = self.env._robot.data.root_ang_vel_b        # (N,3) body rates
        drone_pos_gate  = self.env._pose_drone_wrt_gate              # (N,3) rel to current gate
        prev_actions    = self.env._previous_actions                 # (N,4)

        # Rotation matrix first two columns (6d) instead of quaternion (4d).
        # Avoids quaternion double-cover ambiguity; matches paper 2406.12505 state repr.
        rot_mat = matrix_from_quat(self.env._robot.data.root_quat_w)     # (N,3,3)
        rot_mat_cols12 = rot_mat[:, :, :2].reshape(self.num_envs, 6)     # (N,6)

        drone_pos_w = self.env._robot.data.root_link_pos_w

        # Gate i+1 look-ahead
        next_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]
        drone_pos_next_gate, _ = subtract_frame_transforms(
            self.env._waypoints[next_idx, :3],
            self.env._waypoints_quat[next_idx, :],
            drone_pos_w
        )   # (N,3)

        # Gate i+2 look-ahead: critical for powerloop — after passing gate 2, the policy
        # needs to see gate 3 (also north-approached, same side) to anticipate the loop-back.
        next2_idx = (self.env._idx_wp + 2) % self.env._waypoints.shape[0]
        drone_pos_next2_gate, _ = subtract_frame_transforms(
            self.env._waypoints[next2_idx, :3],
            self.env._waypoints_quat[next2_idx, :],
            drone_pos_w
        )   # (N,3)

        obs = torch.cat([
            drone_lin_vel_b,        # 3
            drone_ang_vel_b,        # 3
            rot_mat_cols12,         # 6  (replaces quat 4d → net +2)
            drone_pos_gate,         # 3
            drone_pos_next_gate,    # 3
            drone_pos_next2_gate,   # 3  (new)
            prev_actions,           # 4
        ], dim=-1)  # total: 25 dims

        # Privileged critic observations: actor obs + normalized dynamics parameters.
        # The critic sees the true randomized dynamics during training (discarded at deployment).
        # Each param divided by its nominal value → ≈1.0 at nominal, within [0.5,2.0] range.
        dyn = torch.stack([
            self.env._thrust_to_weight       / self.env._twr_value,
            self.env._K_aero[:, 0]           / self.env._k_aero_xy_value,
            self.env._K_aero[:, 2]           / self.env._k_aero_z_value,
            self.env._kp_omega[:, 0]         / self.env._kp_omega_rp_value,
            self.env._ki_omega[:, 0]         / self.env._ki_omega_rp_value,
            self.env._kd_omega[:, 0]         / self.env._kd_omega_rp_value,
            self.env._kp_omega[:, 2]         / self.env._kp_omega_y_value,
            self.env._ki_omega[:, 2]         / self.env._ki_omega_y_value,
            self.env._kd_omega[:, 2]         / self.env._kd_omega_y_value,
        ], dim=1)   # (N, 9)

        critic_obs = torch.cat([obs, dyn], dim=-1)  # (N, 34)

        return {"policy": obs, "critic": critic_obs}

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to initial states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # Logging for training mode
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0
            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)
            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)

        # Call robot reset first
        self.env._robot.reset(env_ids)

        # Initialize model paths if needed
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        # Reset action buffers
        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0

        # Reset progress reward buffer (0 → first-step progress clamped to 0, no spurious reward)
        self.env._last_distance_to_approach[env_ids] = 0.0

        # Domain randomization: randomize dynamics on each episode reset to match eval ranges.
        # Aero range is narrowed (0.8–1.2×) vs eval (0.5–2×) to avoid slowing early learning —
        # expand to full range once the policy reliably passes gates.
        if self.cfg.is_train:
            # Thrust-to-weight: ±5% of nominal (matches eval range exactly)
            self.env._thrust_to_weight[env_ids] = (
                torch.empty(n_reset, device=self.device).uniform_(0.95, 1.05) * self.env._twr_value
            )
            # Aerodynamic drag: full TA eval range (0.5–2.0×) for robustness
            self.env._K_aero[env_ids, :2] = (
                torch.empty(n_reset, 1, device=self.device).uniform_(0.5, 2.0).expand(n_reset, 2)
                * self.env._k_aero_xy_value
            )
            self.env._K_aero[env_ids, 2] = (
                torch.empty(n_reset, device=self.device).uniform_(0.5, 2.0) * self.env._k_aero_z_value
            )
            # PID gains roll/pitch: kp/ki ±15%, kd ±30%
            self.env._kp_omega[env_ids, :2] = (
                torch.empty(n_reset, 1, device=self.device).uniform_(0.85, 1.15).expand(n_reset, 2)
                * self.env._kp_omega_rp_value
            )
            self.env._ki_omega[env_ids, :2] = (
                torch.empty(n_reset, 1, device=self.device).uniform_(0.85, 1.15).expand(n_reset, 2)
                * self.env._ki_omega_rp_value
            )
            self.env._kd_omega[env_ids, :2] = (
                torch.empty(n_reset, 1, device=self.device).uniform_(0.7, 1.3).expand(n_reset, 2)
                * self.env._kd_omega_rp_value
            )
            # PID gains yaw: same ranges as roll/pitch
            self.env._kp_omega[env_ids, 2] = (
                torch.empty(n_reset, device=self.device).uniform_(0.85, 1.15) * self.env._kp_omega_y_value
            )
            self.env._ki_omega[env_ids, 2] = (
                torch.empty(n_reset, device=self.device).uniform_(0.85, 1.15) * self.env._ki_omega_y_value
            )
            self.env._kd_omega[env_ids, 2] = (
                torch.empty(n_reset, device=self.device).uniform_(0.7, 1.3) * self.env._kd_omega_y_value
            )

        # Reset joints state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids]

        # Curriculum reset: spawn at a random gate so the policy sees all gates during training
        waypoint_indices = torch.randint(
            0, self.env._waypoints.shape[0], (n_reset,),
            device=self.device, dtype=self.env._idx_wp.dtype
        )

        # get starting poses behind waypoints
        x0_wp = self.env._waypoints[waypoint_indices][:, 0]
        y0_wp = self.env._waypoints[waypoint_indices][:, 1]
        theta = self.env._waypoints[waypoint_indices][:, -1]
        z_wp = self.env._waypoints[waypoint_indices][:, 2]

        x_local = torch.empty(n_reset, device=self.device).uniform_(-2.0, -0.5)
        y_local = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
        z_local = torch.empty(n_reset, device=self.device).uniform_(-0.15, 0.15)

        # rotate local pos to global frame
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local
        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_local + z_wp

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # point drone towards its assigned gate with per-env yaw noise
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise = torch.empty(n_reset, device=self.device).uniform_(-0.3, 0.3)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise
        )
        default_root_state[:, 3:7] = quat

        # Randomize initial linear velocity for robustness to non-zero starting states
        if self.cfg.is_train:
            init_vel = torch.empty(n_reset, 3, device=self.device).uniform_(-0.5, 0.5)
            default_root_state[:, 7:10] = init_vel

        # Handle play mode initial position
        if not self.cfg.is_train:
            # x_local and y_local are randomly sampled
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            # rotate local pos to global frame
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local
            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot
            z0 = 0.05

            # point drone towards the zeroth gate
            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0)
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0
            )
            default_root_state[:, 3:7] = quat
            waypoint_indices = self.env._initial_wp

        # Set waypoint indices and desired positions
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :2] = self.env._waypoints[waypoint_indices, :2].clone()
        self.env._desired_pos_w[env_ids, 2] = self.env._waypoints[waypoint_indices, 2].clone()

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._desired_pos_w[env_ids, :2] - self.env._robot.data.root_link_pos_w[env_ids, :2], dim=1
        )
        self.env._n_gates_passed[env_ids] = 0

        # Write state to simulation
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset variables
        self.env._yaw_n_laps[env_ids] = 0

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3]
        )

        self.env._prev_x_drone_wrt_gate[env_ids] = 1.0

        self.env._crashed[env_ids] = 0