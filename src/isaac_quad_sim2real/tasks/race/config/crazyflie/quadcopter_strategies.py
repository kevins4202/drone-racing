# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional

from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_from_euler_xyz,
    matrix_from_quat,
    quat_mul,
)

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Racing strategy: sign-change gate detection, body-frame observations, domain randomization."""

    def __init__(self, env: QuadcopterEnv):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # ------------------------------------------------------------------ #
        # Domain randomization bounds (per the assignment specification)      #
        # ------------------------------------------------------------------ #
        # cfg.thrust_to_weight is set to 3.15 in QuadcopterEnv.__init__ before
        # this strategy is constructed, so it is safe to reference here.
        if self.cfg.is_train:
            # Wide ranges for training
            self._twr_min = self.cfg.thrust_to_weight * 0.75
            self._twr_max = self.cfg.thrust_to_weight * 1.25
            self._k_aero_xy_min = self.cfg.k_aero_xy * 0.25
            self._k_aero_xy_max = self.cfg.k_aero_xy * 4.0
            self._k_aero_z_min  = self.cfg.k_aero_z  * 0.25
            self._k_aero_z_max  = self.cfg.k_aero_z  * 4.0
            self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.60
            self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.40
            self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.60
            self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.40
            self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.40
            self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.60
            self._kp_omega_y_min = self.cfg.kp_omega_y * 0.60
            self._kp_omega_y_max = self.cfg.kp_omega_y * 1.40
            self._ki_omega_y_min = self.cfg.ki_omega_y * 0.60
            self._ki_omega_y_max = self.cfg.ki_omega_y * 1.40
            self._kd_omega_y_min = self.cfg.kd_omega_y * 0.40
            self._kd_omega_y_max = self.cfg.kd_omega_y * 1.60
        else:
            # Narrow ranges for play/video
            self._twr_min = self.cfg.thrust_to_weight * 0.95
            self._twr_max = self.cfg.thrust_to_weight * 1.05
            self._k_aero_xy_min = self.cfg.k_aero_xy * 0.5
            self._k_aero_xy_max = self.cfg.k_aero_xy * 2.0
            self._k_aero_z_min  = self.cfg.k_aero_z  * 0.5
            self._k_aero_z_max  = self.cfg.k_aero_z  * 2.0
            self._kp_omega_rp_min = self.cfg.kp_omega_rp * 0.85
            self._kp_omega_rp_max = self.cfg.kp_omega_rp * 1.15
            self._ki_omega_rp_min = self.cfg.ki_omega_rp * 0.85
            self._ki_omega_rp_max = self.cfg.ki_omega_rp * 1.15
            self._kd_omega_rp_min = self.cfg.kd_omega_rp * 0.70
            self._kd_omega_rp_max = self.cfg.kd_omega_rp * 1.30
            self._kp_omega_y_min = self.cfg.kp_omega_y * 0.85
            self._kp_omega_y_max = self.cfg.kp_omega_y * 1.15
            self._ki_omega_y_min = self.cfg.ki_omega_y * 0.85
            self._ki_omega_y_max = self.cfg.ki_omega_y * 1.15
            self._kd_omega_y_min = self.cfg.kd_omega_y * 0.70
            self._kd_omega_y_max = self.cfg.kd_omega_y * 1.30

        # tau_m is not in the randomization spec — set fixed once
        self.env._tau_m[:] = self.env._tau_m_value

        # Per-episode metric accumulators (reset in reset_idx)
        self._episode_gate_pass_errors = torch.zeros(self.num_envs, device=self.device)
        self._episode_speed_at_pass    = torch.zeros(self.num_envs, device=self.device)
        self._episode_backwards_count  = torch.zeros(self.num_envs, device=self.device)

        # Episode reward sums for logging
        if self.cfg.is_train and hasattr(env, 'rew'):
            reward_keys = ["progress_goal", "gate_pass", "crash", "cmd"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in reward_keys
            }

    # ---------------------------------------------------------------------- #
    # Domain randomization helper                                             #
    # ---------------------------------------------------------------------- #

    def _randomize_params(self, env_ids: torch.Tensor, randomize: bool):
        """Randomize per-environment physical parameters on reset."""
        n   = len(env_ids)
        dev = self.device

        def u(lo, hi):
            return torch.empty(n, device=dev).uniform_(lo, hi)

        if randomize:
            self.env._thrust_to_weight[env_ids] = u(self._twr_min, self._twr_max)

            self.env._K_aero[env_ids, 0] = u(self._k_aero_xy_min, self._k_aero_xy_max)
            self.env._K_aero[env_ids, 1] = u(self._k_aero_xy_min, self._k_aero_xy_max)
            self.env._K_aero[env_ids, 2] = u(self._k_aero_z_min,  self._k_aero_z_max)

            # Roll/pitch PID — identical gain on axes 0 and 1
            rp_kp = u(self._kp_omega_rp_min, self._kp_omega_rp_max)
            rp_ki = u(self._ki_omega_rp_min, self._ki_omega_rp_max)
            rp_kd = u(self._kd_omega_rp_min, self._kd_omega_rp_max)
            self.env._kp_omega[env_ids, 0] = rp_kp
            self.env._kp_omega[env_ids, 1] = rp_kp
            self.env._ki_omega[env_ids, 0] = rp_ki
            self.env._ki_omega[env_ids, 1] = rp_ki
            self.env._kd_omega[env_ids, 0] = rp_kd
            self.env._kd_omega[env_ids, 1] = rp_kd

            # Yaw PID
            self.env._kp_omega[env_ids, 2] = u(self._kp_omega_y_min, self._kp_omega_y_max)
            self.env._ki_omega[env_ids, 2] = u(self._ki_omega_y_min, self._ki_omega_y_max)
            self.env._kd_omega[env_ids, 2] = u(self._kd_omega_y_min, self._kd_omega_y_max)
        else:
            self.env._thrust_to_weight[env_ids] = self.cfg.thrust_to_weight

            self.env._K_aero[env_ids, 0] = self.cfg.k_aero_xy
            self.env._K_aero[env_ids, 1] = self.cfg.k_aero_xy 
            self.env._K_aero[env_ids, 2] = self.cfg.k_aero_z
            
            rp_kp = self.cfg.kp_omega_rp
            rp_ki = self.cfg.ki_omega_rp
            rp_kd = self.cfg.kd_omega_rp
            self.env._kp_omega[env_ids, 0] = rp_kp
            self.env._kp_omega[env_ids, 1] = rp_kp
            self.env._ki_omega[env_ids, 0] = rp_ki
            self.env._ki_omega[env_ids, 1] = rp_ki
            self.env._kd_omega[env_ids, 0] = rp_kd
            self.env._kd_omega[env_ids, 1] = rp_kd

            # Yaw PID
            self.env._kp_omega[env_ids, 2] = self.cfg.kp_omega_y
            self.env._ki_omega[env_ids, 2] = self.cfg.ki_omega_y
            self.env._kd_omega[env_ids, 2] = self.cfg.kd_omega_y

        # tau_m: not randomized per spec
        self.env._tau_m[env_ids] = self.env._tau_m_value

    # ---------------------------------------------------------------------- #
    # Rewards                                                                 #
    # ---------------------------------------------------------------------- #

    def get_rewards(self) -> torch.Tensor:
        """Compute per-timestep racing rewards.

        Components (arxiv 2406.12505):
          r_prog  — progress toward current gate (Δ-distance, signed)
          r_pass  — sparse gate-pass bonus
          r_crash — per-step contact penalty
          r_cmd   — command magnitude + smoothness penalty
        """
        num_waypoints = self.env._waypoints.shape[0]
        drone_pos_w   = self.env._robot.data.root_link_pos_w        # (N, 3)

        # ------------------------------------------------------------------ #
        # Gate traversal detection via sign change on gate-frame x-axis      #
        # ------------------------------------------------------------------ #
        # _pose_drone_wrt_gate = (drone - gate) in gate local frame
        # Gate x-axis = gate normal.  Front face → x > 0; back face → x < 0.
        # Correct traversal: x crosses + → −
        # Backwards traversal: x crosses − → + (illegal → terminate)
        current_x = self.env._pose_drone_wrt_gate[:, 0]
        current_y = self.env._pose_drone_wrt_gate[:, 1]
        current_z = self.env._pose_drone_wrt_gate[:, 2]
        prev_x    = self.env._prev_x_drone_wrt_gate

        half_side     = 0.5   # gate opening is 1 m × 1 m
        within_bounds = (torch.abs(current_y) < half_side) & (torch.abs(current_z) < half_side)
        gate_passed   = (prev_x > 0) & (current_x <= 0) & within_bounds
        backwards     = (prev_x < 0) & (current_x >= 0) & within_bounds

        ids_gate_passed = torch.where(gate_passed)[0]
        ids_backwards   = torch.where(backwards)[0]

        # Advance waypoint index and counters for envs that passed a gate
        if len(ids_gate_passed) > 0:
            self.env._idx_wp[ids_gate_passed] = (
                self.env._idx_wp[ids_gate_passed] + 1
            ) % num_waypoints
            self.env._n_gates_passed[ids_gate_passed] += 1

            # Update desired position to new gate
            new_idx = self.env._idx_wp[ids_gate_passed]
            self.env._desired_pos_w[ids_gate_passed, :3] = self.env._waypoints[new_idx, :3]

            # Update last-distance for progress reward (reset to distance to NEW gate)
            new_gate_pos = self.env._waypoints[new_idx, :3]
            self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(
                new_gate_pos - drone_pos_w[ids_gate_passed], dim=1
            )

            # Set prev_x relative to NEW gate (so detection works immediately next step)
            new_gate_quat = self.env._waypoints_quat[new_idx, :]
            new_pose, _ = subtract_frame_transforms(
                new_gate_pos, new_gate_quat, drone_pos_w[ids_gate_passed]
            )
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pose[:, 0]

            # Accumulate WandB metrics at gate passage
            gate_pass_err = torch.sqrt(current_y[ids_gate_passed] ** 2 +
                                       current_z[ids_gate_passed] ** 2)
            speed_now = torch.linalg.norm(
                self.env._robot.data.root_com_lin_vel_b[ids_gate_passed], dim=1
            )
            self._episode_gate_pass_errors[ids_gate_passed] += gate_pass_err
            self._episode_speed_at_pass[ids_gate_passed]    += speed_now

        # Update prev_x for envs that did NOT pass a gate this step
        not_passed_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        not_passed_mask[ids_gate_passed] = False
        self.env._prev_x_drone_wrt_gate[not_passed_mask] = current_x[not_passed_mask]

        # Handle backwards traversal — force termination next step
        if len(ids_backwards) > 0:
            self.env._crashed[ids_backwards] = 200   # exceeds _get_dones threshold of 100
            self._episode_backwards_count[ids_backwards] += 1.0

        # ------------------------------------------------------------------ #
        # r_prog: progress reward (arxiv 2406.12505, λ₁ = 0.5)              #
        # r_prog = d_{t-1} - d_t  (positive = moved closer to gate)         #
        # ------------------------------------------------------------------ #
        dist_to_gate = torch.linalg.norm(
            self.env._waypoints[self.env._idx_wp, :3] - drone_pos_w, dim=1
        )
        r_prog = self.env._last_distance_to_goal - dist_to_gate
        # Clamp to avoid large negative spike when target gate switches
        r_prog = torch.clamp(r_prog, min=-0.5, max=5.0)
        self.env._last_distance_to_goal = dist_to_gate.clone()

        # ------------------------------------------------------------------ #
        # r_pass: sparse gate-pass bonus                                      #
        # ------------------------------------------------------------------ #
        r_pass = gate_passed.float()

        # ------------------------------------------------------------------ #
        # r_crash: per-step contact penalty + crash accumulator               #
        # ------------------------------------------------------------------ #
        contact_forces = self.env._contact_sensor.data.net_forces_w  # (N, 1, 3)
        crashed_now    = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).int()
        mask           = (self.env.episode_length_buf > 100).int()
        self.env._crashed = self.env._crashed + crashed_now * mask
        r_crash = crashed_now.float()

        # ------------------------------------------------------------------ #
        # r_cmd: command magnitude + smoothness penalty (arxiv 2406.12505)   #
        # r_cmd = 0.0005*||a|| + 0.0002*||Δa||²                             #
        # ------------------------------------------------------------------ #
        cmd_mag    = torch.linalg.norm(self.env._actions, dim=1)
        cmd_smooth = torch.linalg.norm(
            self.env._actions - self.env._previous_actions, dim=1
        ) ** 2
        r_cmd = 0.0005 * cmd_mag + 0.0002 * cmd_smooth

        if self.cfg.is_train:
            rewards = {
                "progress_goal": r_prog  * self.env.rew['progress_goal_reward_scale'],
                "gate_pass":     r_pass  * self.env.rew['gate_pass_reward_scale'],
                "crash":         r_crash * self.env.rew['crash_reward_scale'],
                "cmd":           r_cmd   * self.env.rew['cmd_reward_scale'],
            }
            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            # Apply death cost on episode termination (crash, altitude, backwards)
            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew['death_cost'],
                reward,
            )

            # Logging
            for key, value in rewards.items():
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    # ---------------------------------------------------------------------- #
    # Observations                                                            #
    # ---------------------------------------------------------------------- #

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """20-dim observation vector, all position/velocity in drone body frame.

        Layout:
          v_b         (3)  body-frame linear velocity
          omega_b     (3)  body-frame angular velocity
          delta_p0_b  (3)  vector to target gate in body frame
          delta_p1_b  (3)  vector to next gate in body frame  ← power-loop lookahead
          q_rel       (4)  quaternion: gate_inv * drone
          prev_action (4)  previous action
        """
        drone_pos_w   = self.env._robot.data.root_link_pos_w       # (N, 3)
        drone_quat_w  = self.env._robot.data.root_quat_w            # (N, 4) [w,x,y,z]
        v_b           = self.env._robot.data.root_com_lin_vel_b     # (N, 3) already body frame
        omega_b       = self.env._robot.data.root_ang_vel_b         # (N, 3) already body frame

        # Rotation matrix: body → world; transpose = world → body
        R_wb = matrix_from_quat(drone_quat_w)        # (N, 3, 3)
        R_bw = R_wb.transpose(-1, -2)                # (N, 3, 3)

        # Target gate (current waypoint)
        gate0_pos_w = self.env._waypoints[self.env._idx_wp, :3]     # (N, 3)
        delta0_w    = gate0_pos_w - drone_pos_w                      # (N, 3)
        delta_p0_b  = torch.bmm(R_bw, delta0_w.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # Next gate (lookahead — critical for power loop between gates 2 and 3)
        num_wp      = self.env._waypoints.shape[0]
        next_idx    = (self.env._idx_wp + 1) % num_wp
        gate1_pos_w = self.env._waypoints[next_idx, :3]             # (N, 3)
        delta1_w    = gate1_pos_w - drone_pos_w                      # (N, 3)
        delta_p1_b  = torch.bmm(R_bw, delta1_w.unsqueeze(-1)).squeeze(-1)  # (N, 3)

        # Relative orientation: q_rel = q_gate_inv * q_drone
        gate_quat    = self.env._waypoints_quat[self.env._idx_wp, :]  # (N, 4)
        gate_quat_inv = gate_quat.clone()
        gate_quat_inv[:, 1:] *= -1                                     # conjugate = inverse for unit quat
        q_rel = quat_mul(gate_quat_inv, drone_quat_w)                  # (N, 4)

        # Previous action
        prev_actions = self.env._previous_actions                      # (N, 4)

        obs = torch.cat([v_b, omega_b, delta_p0_b, delta_p1_b, q_rel, prev_actions], dim=-1)
        # Shape: (N, 3+3+3+3+4+4) = (N, 20)

        return {"policy": obs}

    # ---------------------------------------------------------------------- #
    # Reset                                                                   #
    # ---------------------------------------------------------------------- #

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments.

        Key features vs original code:
         - Random starting gate (curriculum via exposure to all course segments)
         - Random starting position (0.5–3 m behind gate, ±0.4 m lateral, ±0.15 m height)
         - Random starting velocity (0–2 m/s toward gate center)
         - Domain randomization of physical parameters per environment
         - WandB metric logging
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        # ------------------------------------------------------------------ #
        # WandB / logging                                                     #
        # ------------------------------------------------------------------ #
        if self.cfg.is_train and hasattr(self, '_episode_sums'):
            extras = dict()

            # Reward component sums
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0

            # Termination breakdown
            extras["Episode_Termination/died"]     = torch.count_nonzero(
                self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(
                self.env.reset_time_outs[env_ids]).item()

            # Racing-specific metrics
            gates      = self.env._n_gates_passed[env_ids].float()
            num_wp     = self.env._waypoints.shape[0]
            pass_count = gates.clamp(min=1)

            extras["Episode_Metric/gates_passed_mean"]    = gates.mean().item()
            extras["Episode_Metric/gates_passed_max"]     = gates.max().item()
            extras["Episode_Metric/laps_completed_mean"]  = (gates / num_wp).mean().item()

            gate_err   = self._episode_gate_pass_errors[env_ids] / pass_count
            speed_pass = self._episode_speed_at_pass[env_ids]    / pass_count
            extras["Episode_Metric/gate_pass_error_mean"]    = gate_err.mean().item()
            extras["Episode_Metric/speed_at_gate_pass_mean"] = speed_pass.mean().item()
            extras["Episode_Metric/backwards_traversal_mean"] = (
                self._episode_backwards_count[env_ids].mean().item()
            )

            self.env.extras["log"] = extras

        # ------------------------------------------------------------------ #
        # Reset robot physics state                                           #
        # ------------------------------------------------------------------ #
        self.env._robot.reset(env_ids)

        # Initialize gate model paths (first time only)
        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [
                f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)
            ]
            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)
            self.env._models_paths_initialized = True

        n_reset = len(env_ids)

        # Stagger episode lengths on the very first full reset to prevent synchronized resets
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # ------------------------------------------------------------------ #
        # Zero action / controller state buffers                             #
        # ------------------------------------------------------------------ #
        self.env._actions[env_ids]               = 0.0
        self.env._previous_actions[env_ids]      = 0.0
        self.env._previous_yaw[env_ids]          = 0.0
        self.env._motor_speeds[env_ids]          = 0.0
        self.env._previous_omega_meas[env_ids]   = 0.0
        self.env._previous_omega_err[env_ids]    = 0.0
        self.env._omega_err_integral[env_ids]    = 0.0

        # Reset joint state
        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # ------------------------------------------------------------------ #
        # Domain randomization                                                #
        # ------------------------------------------------------------------ #
        self._randomize_params(env_ids, self.cfg.randomize_domain)

        # ------------------------------------------------------------------ #
        # Reset per-episode metric accumulators                               #
        # ------------------------------------------------------------------ #
        self._episode_gate_pass_errors[env_ids] = 0.0
        self._episode_speed_at_pass[env_ids]    = 0.0
        self._episode_backwards_count[env_ids]  = 0.0

        # ------------------------------------------------------------------ #
        # Initial state                                                       #
        # ------------------------------------------------------------------ #
        default_root_state = self.env._robot.data.default_root_state[env_ids].clone()

        num_waypoints = self.env._waypoints.shape[0]

        if self.cfg.is_train:
            # Random gate: expose policy to all parts of the course
            waypoint_indices = torch.randint(
                0, num_waypoints, (n_reset,), device=self.device, dtype=self.env._idx_wp.dtype
            )
        else:
            waypoint_indices = torch.full(
                (n_reset,), self.env._initial_wp, device=self.device, dtype=self.env._idx_wp.dtype
            )

        # Gate reference data for selected waypoints
        x0_wp = self.env._waypoints[waypoint_indices, 0]
        y0_wp = self.env._waypoints[waypoint_indices, 1]
        z_wp  = self.env._waypoints[waypoint_indices, 2]
        theta = self.env._waypoints[waypoint_indices, -1]   # gate yaw in world

        # ---- Starting position in gate-local frame ----
        # x_local < 0: drone starts "behind" the gate in its local coordinate.
        # After rotation to world frame, gate-frame x = +|x_local| > 0, meaning
        # the drone is on the front face of the gate (correct approach side).
        if self.cfg.is_train:
            x_local = -torch.empty(n_reset, device=self.device).uniform_(0.5, 3.0)
            y_local =  torch.empty(n_reset, device=self.device).uniform_(-0.4, 0.4)
            z_local =  torch.empty(n_reset, device=self.device).uniform_(-0.4, 0.4)

            # Gate 3: horizontal power loop — spawn east of gate (loop arcs south→east→north→south)
            # y_local > 0 maps to initial_x = gate_x + y_local (east in world), which is
            # where the drone is mid-loop before its final southward approach to gate 3.
            # is_gate3 = (waypoint_indices == 3)
            # if is_gate3.any():
            #     z_local[is_gate3] = torch.empty(is_gate3.sum().item(), device=self.device).uniform_(0.75, 2.0)
        else:
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0,  1.0)
            z_local = torch.zeros(1, device=self.device)

        # Rotate gate-local (x_local, y_local) to world frame offset from gate centre
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x_rot = cos_theta * x_local - sin_theta * y_local
        y_rot = sin_theta * x_local + cos_theta * y_local

        initial_x = x0_wp - x_rot
        initial_y = y0_wp - y_rot
        initial_z = z_wp  + z_local

        default_root_state[:, 0] = initial_x
        default_root_state[:, 1] = initial_y
        default_root_state[:, 2] = initial_z

        # ---- Starting heading: toward gate center + small yaw noise ----
        initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
        yaw_noise   = torch.empty(n_reset, device=self.device).uniform_(-0.2, 0.2)
        quat = quat_from_euler_xyz(
            torch.zeros(n_reset, device=self.device),
            torch.zeros(n_reset, device=self.device),
            initial_yaw + yaw_noise,
        )
        default_root_state[:, 3:7] = quat

        # ---- Starting velocity: random speed toward gate center ----
        gate_pos_tgt = self.env._waypoints[waypoint_indices, :3]  # (N, 3)
        start_pos    = torch.stack([initial_x, initial_y, initial_z], dim=1)
        direction    = gate_pos_tgt - start_pos
        direction    = direction / (torch.linalg.norm(direction, dim=1, keepdim=True) + 1e-6)
        if self.cfg.is_train:
            speed = torch.empty(n_reset, device=self.device).uniform_(0.0, 5.0).unsqueeze(1)
        else:
            speed = torch.zeros(n_reset, 1, device=self.device)
        initial_vel = direction * speed          # (N, 3)

        default_root_state[:, 7:10]  = initial_vel
        default_root_state[:, 10:13] = 0.0       # zero angular velocity

        # ------------------------------------------------------------------ #
        # Write to sim                                                        #
        # ------------------------------------------------------------------ #
        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # ------------------------------------------------------------------ #
        # Waypoint bookkeeping                                                #
        # ------------------------------------------------------------------ #
        self.env._idx_wp[env_ids] = waypoint_indices

        self.env._desired_pos_w[env_ids, :3] = self.env._waypoints[waypoint_indices, :3].clone()

        # _last_distance_to_goal: 3-D distance from start to target gate
        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            gate_pos_tgt - start_pos, dim=1
        )

        self.env._n_gates_passed[env_ids] = 0
        self.env._yaw_n_laps[env_ids]     = 0
        self.env._crashed[env_ids]        = 0

        # ------------------------------------------------------------------ #
        # _prev_x_drone_wrt_gate                                              #
        # Sign convention: x_local < 0 → gate_frame_x ≈ +|x_local| > 0      #
        # Using -x_local directly avoids stale sim readback issues            #
        # (root_link_state_w may not reflect the newly written position yet). #
        # ------------------------------------------------------------------ #
        self.env._prev_x_drone_wrt_gate[env_ids] = -x_local  # guaranteed positive

        # Also initialise _pose_drone_wrt_gate (best-effort from sim, used in obs on step 1)
        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            self.env._robot.data.root_link_state_w[env_ids, :3],
        )
