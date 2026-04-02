# Autonomous Drone Racing via Proximal Policy Optimization

**Kevin Song · ESE 651 · Isaac Lab / Crazyflie 2.x · April 2026**

---

## 1. Problem Formulation

The task is to train a quadrotor to autonomously navigate a closed gate course as fast as possible, completing laps without crashing. The policy is trained entirely in simulation using massively parallel reinforcement learning (4,096 environments) and transfers to hardware through domain randomization. The track used is the **powerloop** layout — seven gates arranged in a figure-eight with altitude changes from 0.75 m to 2.0 m — chosen for its directional reversals and loop maneuver that stress both agility and long-horizon planning.

The MDP is defined as follows. At each policy step (50 Hz), the agent observes a 20-dimensional state vector, outputs a 4-dimensional continuous action, and receives a scalar reward. Episodes terminate upon crash, altitude violation, backwards gate traversal, or timeout at 30 seconds.

---

## 2. Dynamics and Action Space

### Motor and Aerodynamic Model

The simulation runs at 500 Hz. Each motor is modeled with a first-order lag (τ_m = 0.005 s) to capture spin-up dynamics:

```
ω̇ᵢ = (ω_des,i − ωᵢ) / τ_m
```

Thrust and torque per rotor follow standard quadrotor equations: `Fᵢ = k_η · ωᵢ²` and `Mᵢ = k_m · ωᵢ²`. A 4×4 allocation matrix maps individual rotor forces to collective thrust and body moments. Aerodynamic drag is modeled as velocity-proportional drag scaled by total rotor speed:

```
F_drag = −(Σωᵢ) · K_aero ⊙ v_b
```

where `K_aero = [k_xy, k_xy, k_z]` with nominal values 9.18×10⁻⁷ and 10.3×10⁻⁷ respectively.

### Collective Thrust and Body Rate (CTBR) Interface

The policy outputs a 4-dimensional action `a = [T̃, ṗ_des, q̃_des, r̃_des]` clipped to [−1, 1]. Collective thrust is decoded linearly from [−1,1] to [0, TWR·mg]. Desired body rates are scaled to ±100°/s for roll/pitch and ±200°/s for yaw. A 500 Hz PID controller closes the loop on body rate error:

```
M_cmd = I · (Kₚ·eω + Kᵢ·∫eω − K_d·ω̇_meas)
```

Roll/pitch gains: Kₚ = 250, Kᵢ = 500, K_d = 2.5. Yaw gains: Kₚ = 120, Kᵢ = 16.7, K_d = 0. This interface decouples high-level trajectory planning (50 Hz policy) from low-level attitude stabilization (500 Hz PID), matching the real Crazyflie firmware architecture and simplifying the learned action space.

---

## 3. Observation Space

The 20-dimensional observation vector is constructed entirely in the drone's body frame, making the policy invariant to global position and heading. No global pose is provided, forcing the network to learn relative geometric reasoning.

| Component | Dim | Description |
|-----------|-----|-------------|
| `v_b` | 3 | Body-frame linear velocity (m/s) |
| `ω_b` | 3 | Body-frame angular velocity (rad/s) |
| `Δp₀_b` | 3 | Vector from drone to current target gate, in body frame |
| `Δp₁_b` | 3 | Vector from drone to **next** gate in body frame — lookahead for turn anticipation |
| `q_rel` | 4 | Relative quaternion: q_gate⁻¹ ⊗ q_drone (alignment with gate normal) |
| `a_prev` | 4 | Previous action (enables smoothness reasoning) |

The lookahead gate vector `Δp₁_b` is critical for the power loop segment, where the drone must begin turning before it fully clears the current gate. Without it, the policy lacks the geometric context needed to pre-rotate. The relative orientation quaternion `q_rel` provides gate-alignment feedback, rewarding approaches that are perpendicular to the gate face.

---

## 4. Reward Function

The reward function follows the formulation of Kaufmann et al. (arXiv 2406.12505) with tuned scales. Four components are summed per timestep, plus a terminal penalty.

| Component | Formula | Scale | Purpose |
|-----------|---------|-------|---------|
| Progress `r_prog` | `d_{t−1} − d_t`, clipped to [−0.5, 5.0] | +1.5 | Dense guidance toward the active gate |
| Gate pass `r_pass` | 1 on valid traversal, else 0 (sparse) | +5.0 | Primary racing objective per gate |
| Crash `r_crash` | 1 if contact force > 0 (per step) | −0.5 | Accumulating penalty for sustained contact |
| Command `r_cmd` | `0.0005‖a‖ + 0.0002‖Δa‖²` | −0.5 | Penalizes large and jerky control inputs |
| Death penalty | Applied at episode termination | −10.0 | Discourages risky maneuvers leading to crash |

**Scale rationale.** The gate-pass bonus (5.0) dominates over accumulated progress (~2–3 per lap at 2 m/s), so the policy's primary incentive is gate traversal rather than simply flying fast toward gates. The crash penalty accumulates per step over a 100-step grace window, punishing sustained contact more harshly than brief rim touches during aggressive lines. The command cost discourages actuator saturation and produces smoother trajectories that generalize better to hardware.

### Gate Detection: Sign-Change Method

Each gate has a local coordinate frame where the x-axis is the gate normal. The drone's position is transformed into this frame every step. A valid gate traversal is detected when:

```
prev_x > 0  ∧  curr_x ≤ 0  ∧  |curr_y| < 0.5  ∧  |curr_z| < 0.5
```

This ensures the drone crosses the gate front-to-back within the 1 m × 1 m opening. Backwards traversal (x crosses − → +) is detected symmetrically and triggers immediate episode termination by forcing the crash counter to 200 (above the 100-step threshold). On a valid pass, the waypoint index advances modulo the number of gates, and the progress baseline distance resets to the new gate — preventing a large negative spike in `r_prog` at the moment of gate switch.

---

## 5. PPO Training Configuration

### Network Architecture

| | Architecture | Activation |
|--|--|--|
| **Actor** | MLP [128, 128] | ELU |
| **Critic** | MLP [512, 256, 128, 128] | ELU |

Initial action noise std: 1.0 (anneals to 0 over training). The critic is intentionally larger than the actor: its job is to accurately estimate value under high variance from random starts and parameter variation, while the actor only needs to encode a compact 20-dim → 4-dim mapping.

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Steps per env per update | 24 |
| Mini-batches | 4 |
| Learning epochs per update | 5 |
| Clip ε | 0.2 |
| KL divergence target | 0.01 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Learning rate | 5×10⁻⁴ (adaptive) |
| Value loss coefficient | 1.0 |
| Entropy coefficient | 0.0 |
| Gradient clip norm | 1.0 |

With 4,096 parallel environments and 24 steps per env, each PPO update batch contains **4,096 × 24 = 98,304 transitions**, split into 4 mini-batches of ~24,576 each, processed over 5 gradient epochs. At 50 Hz policy rate this corresponds to approximately 0.48 s of simulated experience per env per update — short enough to keep GAE bias low on a 30 s episode. The adaptive learning rate schedule rescales whenever the mean KL divergence exceeds 0.01, providing stability during the rapid early learning phase.

---

## 6. Domain Randomization and Reset Curriculum

### Domain Randomization

Physical parameters are independently randomized per environment at every episode reset to bridge the sim-to-real gap.

| Parameter | Nominal | Training Range |
|-----------|---------|----------------|
| Thrust-to-weight ratio | 3.15 | [0.75×, 1.25×] |
| Aerodynamic drag k_xy | 9.18×10⁻⁷ | [0.25×, 4.0×] |
| Aerodynamic drag k_z | 10.3×10⁻⁷ | [0.25×, 4.0×] |
| Rate PID Kₚ (roll/pitch, yaw) | — | [0.60×, 1.40×] |
| Rate PID Kᵢ (roll/pitch, yaw) | — | [0.60×, 1.40×] |
| Rate PID K_d (roll/pitch, yaw) | — | [0.40×, 1.60×] |

Motor time constant τ_m is held fixed (not in the randomization spec) to avoid destabilizing low-level dynamics during early training. Aerodynamic drag is randomized over a 16× range to cover both near-hover low-drag and aggressive-flight high-drag regimes — the largest uncertainty in real-world flight.

### Reset Curriculum

Each episode independently samples a starting gate uniformly at random from all seven waypoints. This exposes the policy to every segment of the course from the very first iteration, preventing the common failure mode where a policy learns only the first few gates and never encounters the rest. Starting pose is randomized as follows:

- **Behind gate**: 0.5–3.0 m (sampled in gate-local frame, rotated to world frame)
- **Lateral offset**: ±1.0 m
- **Height offset**: ±0.4 m
- **Initial heading**: toward gate center ± 0.2 rad yaw noise
- **Initial speed**: 0–5 m/s along the gate approach direction

The high initial speed range (up to 5 m/s) forces the policy to recover from fast-approach conditions, which are common mid-race after clearing a gate. On the first full reset at startup, episode lengths are staggered uniformly to prevent synchronized resets throughout training.

---

## 7. Strategy Design: The Power Loop

The powerloop track includes segments (gates 2→3 and 5→6) where successive gates face the same lateral direction and the drone must complete a tight reversal at altitude. The second gate lookahead `Δp₁_b` was specifically added to give the policy early warning of this turn. Without it, the policy has no information about the next gate until the current one is passed, and must learn to overshoot and correct — which is slow. With the lookahead, the network can learn to begin banking before crossing the current gate, matching the behavior of championship-level pilots who pre-rotate into apex.

Gate traversal correctness is enforced by the sign-change detector rather than proximity thresholds: the policy must physically fly *through* the gate opening (within the 1 m frame) in the correct direction. This prevents trivially high-reward behaviors such as circling around the gate.

---

## 8. Summary

The racing strategy combines a dense-sparse hybrid reward (progress + gate pass), a compact 20-dim body-frame observation with next-gate lookahead, a CTBR action interface decoupled from a 500 Hz body-rate PID, and PPO with wide domain randomization and a random-gate reset curriculum. The architecture choices — asymmetric actor/critic, body-frame invariant observations, sign-change gate detection — are motivated by the physical structure of the racing task and the sim-to-real transfer requirement.
