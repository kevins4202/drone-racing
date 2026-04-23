# PPO Update

For each mini-batch, the actor is called again on the stored observations to get log probabilities under the current policy, and the critic is called to get current value estimates. This re-evaluation is necessary because the parameters have changed since the rollout was collected.

Advantage normalization is applied per mini-batch when enabled, zero-centering and scaling the advantages by their standard deviation. This stabilizes training by preventing batches with unusually large returns from dominating the gradient.

The adaptive learning rate schedule computes the KL divergence between the old policy (stored mean and std from the rollout) and the current policy using the closed-form Gaussian KL. If the mean KL exceeds twice the target of 0.01, the learning rate is divided by 1.5. If it falls below half the target, it is multiplied by 1.5. This keeps update sizes consistent without requiring a fixed step size.

The surrogate loss takes the maximum of the negated unclipped and clipped policy gradient objectives. The ratio of new to old log probabilities is computed, and the advantage-weighted ratio is clipped to the range 1 minus epsilon to 1 plus epsilon (epsilon = 0.2). Taking the max of the two negated terms gives the pessimistic bound that is the core of the PPO constraint.

The value loss uses clipped value targets: the predicted value is clipped to within epsilon of the old value estimate before computing the squared error against the discounted return. The max of the clipped and unclipped value losses is taken. The total loss is surrogate plus value loss times 1.0, minus entropy times 0.0. Gradients are clipped to norm 1.0 before the optimizer step.

---

## Reward Function

The reward has five components summed each timestep, plus a death penalty applied at termination.

**Progress reward (scale +1.5).** Each step, the reward includes the change in distance to the current target gate: how much closer the drone got compared to the previous step. This is clipped between -0.5 and 5.0. The clip prevents a large negative spike when the target gate switches right after a pass, since the distance baseline resets to the new gate which may be farther away. This component is dense and gives the policy continuous guidance toward each gate.

**Gate pass reward (scale +5.0).** A sparse bonus of 1.0 is given exactly when the drone successfully passes through a gate in the correct direction. The scale of 5.0 means completing a gate is worth more than any amount of accumulated progress toward it, so the policy is pushed to actually fly through gates rather than just approach them.

**Time penalty (scale -0.01).** A constant -1 per step is applied at every timestep. The progress reward is speed-invariant: the total distance change approaching a gate sums to roughly the same value regardless of speed. Without this, the policy has no incentive to go fast once it can navigate gates reliably. At scale -0.01, flying twice as fast through a gate saves roughly 1.0 reward, which is 20% of a single gate pass bonus.

**Crash penalty (scale -0.5).** A per-step penalty of 1.0 is applied whenever the contact sensor detects nonzero force on the drone. The crash accumulator only starts counting after 100 steps into an episode, to avoid penalizing brief rim grazes during aggressive lines through early training. Once the count exceeds 100, the episode terminates. Total crash cost over a sustained crash is approximately -50 plus the death penalty.

**Command cost (scale -1.0).** The raw penalty is 0.0005 times the action magnitude plus 0.0002 times the squared action change. This discourages large, jerky outputs and produces smoother trajectories that generalize better to hardware.

**Death penalty (-10.0).** Applied at the final step of any episode ending in crash, altitude violation, or backwards gate traversal. This discourages high-risk behavior that terminates episodes early.

---

## Gate Detection

The boilerplate detected gate passes by checking if the drone was within 0.1 meters of the gate center, which cannot enforce correct direction or that the drone flies through the opening. I replaced it with a sign-change method.

Each gate has a local coordinate frame where the x-axis points along the gate normal. The drone's position is projected into this frame every step. A correct traversal is detected when the x-coordinate was positive last step (drone on the approach side) and is zero or negative this step (drone has crossed through), and the y and z offsets are both within 0.5 meters of center. This enforces that the drone flies through the 1 by 1 meter opening rather than around it.

A backwards traversal is detected symmetrically: x going from negative to positive within bounds. This immediately forces episode termination by setting the crash counter above the threshold.

When a gate is passed, the waypoint index advances to the next gate, the progress baseline distance resets to the distance to the new gate, and the sign-change tracker is updated relative to the new gate's frame. This baseline reset is what prevents the progress reward from spiking negatively at the moment of gate switch.

---

## Observations

The final observation vector is 20-dimensional and built entirely in the drone's body frame: body-frame linear velocity (3), body-frame angular velocity (3), the vector from the drone to the current target gate in body frame (3), the vector from the drone to the next gate in body frame (3), the relative quaternion between the current gate orientation and the drone orientation (4), and the previous action (4).

Keeping everything in body frame makes the policy invariant to where on the course the drone is. The same network weights apply at every gate and every position on the track, which helps generalization. The next-gate lookahead vector is the key addition for the powerloop course.

The relative orientation quaternion tells the policy how well it is aligned with the gate face, which is useful for approaching perpendicular to the gate rather than at a skewed angle. The previous action lets the policy reason about smoothness directly, reinforcing the command cost penalty.

---

## Reset Curriculum

The starting gate is chosen uniformly at random from all seven waypoints. This exposes the policy to every segment of the course from the very first iteration, preventing the failure mode where the policy learns the first few gates well but never encounters the rest. Starting position is sampled 0.5 to 3 meters behind the gate in gate-local coordinates, with up to 1 meter of lateral offset and 0.4 meters of height variation. Initial heading points toward the gate center with up to 0.2 radians of yaw noise. Initial speed is sampled uniformly from 0 to 5 m/s along the approach direction, which forces the policy to handle fast-approach conditions that occur naturally mid-race after clearing a gate.

---

## Domain Randomization

Physical parameters are sampled independently per environment on each episode reset, and during training the possible ranges for the values were expanded to cover many possible cases.

---

## Development Progression

**Version 1.** The initial reward design had a gate pass bonus of 100.0, a progress reward of +1.0 toward an approach waypoint (the gate position offset 1 meter along the gate normal), a crash penalty of -1.0 per step, a command cost of -0.001, and a death cost of -50.0. The approach waypoint idea was that directing the drone toward a point in front of the gate on the correct side would naturally enforce approach direction. The progress reward was clamped at zero from below so the drone would not be penalized for the detours. The observation used the drone-to-gate vector in gate frame rather than body frame, and included the world-frame quaternion rather than a relative gate quaternion. Domain randomization ranges were very narrow and minimal.

**Version 2.** The reward scales were adjusted: gate pass brought down to 5.0 from 100.0, progress scale raised to 1.5, crash scale set to -0.5, death cost reduced to -10.0. The progress reward changed from approach-waypoint distance to direct gate distance delta, removing the gate-normal offset. The observation switched from gate-frame position to fully body-frame position and velocity, added angular velocity, and replaced the world quaternion with the relative gate quaternion. Domain randomization ranges were widened to the final values: drag 0.25x to 4x, PID gains 40% off nominal.

**Time penalty** I wanted it to go faster so I added a constant -0.01 per step was added as a time penalty, but removed it since it wasn't successful 100% of the time. 

**Reset curriculum tuning.** I tried to add specific spawn logic for the power loop, first varying z offset to cover the both a horizontal and vertical arc. However, I removed both since the general lateral and height randomization already covered those states and the gate-specific logic added complexity without clear benefit.
