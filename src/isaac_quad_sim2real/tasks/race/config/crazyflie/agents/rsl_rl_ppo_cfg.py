# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rl_cfg import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadcopterPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48          # was 24; longer rollouts → better GAE on 30s episodes
    max_iterations = 1000           # ~1-2 h daytime; pass --max_iterations 3000 for overnight
    save_interval = 100             # was 50
    experiment_name = "quadcopter_direct"
    empirical_normalization = True  # was False; running mean/std norm stabilises training
    wandb_project = "kevin-policy"  # Wandb project name for logging
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256],       # was [128, 128]; more capacity for 25d obs
        critic_hidden_dims=[512, 512, 256], # updated for 34d privileged critic obs
        activation="elu",
        min_std=0.0,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,         # was 0.0; small bonus prevents premature convergence
        num_learning_epochs=5,
        num_mini_batches=8,         # was 4; doubled to match doubled num_steps_per_env
        learning_rate=3.0e-4,       # was 5e-4; paper uses 3e-4
        schedule="adaptive",
        gamma=0.995,                # was 0.99; 0.995^1500≈5e-4 vs 0.99^1500≈2e-7
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
