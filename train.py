import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from torcs_env import TorcsEnv


import torch
torch.cuda.set_device(0)

# ================================================================
# CONFIGURATION — change parameters here without touching rest of code
# ================================================================
# Mode: 'scratch' = SAC from scratch | 'finetune' = after BC / old checkpoint
TRAIN_MODE = 'scratch'  # 'scratch' or 'finetune

CONFIG = {
  # Environment
    'port': 3001,

    # Training
    'total_timesteps': 1_000_000,
    # scratch: ~10k random steps before gradient starts
    # finetune: 0 if buffer pre-filled with expert_data (see block below)
    'learning_starts': 10_000,

    # SAC — common
    'batch_size': 256,
    'gamma': 0.99,           # TORCS episodes long; optionally 0.995 with progress reward
    'tau': 0.005,
    'train_freq': 1,
    'gradient_steps': 1,     # TORCS slow — can use 2–4 for better buffer sampling
    'policy_kwargs': dict(net_arch=[256, 256]),

    # SAC — scratch (from scratch, without BC)
    'scratch': {
        'learning_rate': 3e-4,
        'buffer_size': 300_000,
        'ent_coef': 'auto',      # SB3 default; high exploration at start
        'learning_starts': 10_000,
    },

    # SAC — finetune (after BC / sac_bc_pretrained; less slalom from entropy)
    'finetune': {
        'learning_rate': 1e-4,   # 3e-5 if policy still "drifts"
        'buffer_size': 300_000,
        'ent_coef': 0.05,        # crucial: 'auto' likes to return to noise in turns
        'learning_starts': 0,
        'gradient_steps': 2,
    },

    # Saving
    'save_dir': 'models',
    'log_dir': 'logs',
    'save_freq': 100_000,
    'expert_data_path': 'expert_data.pkl',
    'prefill_expert_buffer': True,  # finetune: keep expert trajectories in buffer
}


def sac_hyperparams():
    """Merges CONFIG + scratch/finetune profile."""
    profile = CONFIG[TRAIN_MODE]
    return {
        'learning_rate': profile['learning_rate'],
        'buffer_size': profile['buffer_size'],
        'batch_size': CONFIG['batch_size'],
        'gamma': CONFIG['gamma'],
        'tau': CONFIG['tau'],
        'train_freq': CONFIG['train_freq'],
        'gradient_steps': profile.get('gradient_steps', CONFIG['gradient_steps']),
        'learning_starts': profile.get('learning_starts', CONFIG['learning_starts']),
        'ent_coef': profile['ent_coef'],
        'policy_kwargs': CONFIG['policy_kwargs'],
    }


def apply_finetune_overrides(model):
    """After SAC.load() saved hyperparameters override CONFIG — set manually."""
    if TRAIN_MODE != 'finetune':
        return
    hp = sac_hyperparams()
    model.learning_rate = hp['learning_rate']
    model.ent_coef = hp['ent_coef']
    model.gradient_steps = hp['gradient_steps']
    model.learning_starts = hp['learning_starts']
    # buffer_size from checkpoint doesn't change after load — new model or larger buffer from scratch

def make_env():
    """Creates and wraps environment in Monitor (logs episodes)."""
    env = TorcsEnv(port=CONFIG['port'])

    # Monitor saves rewards and episode lengths to CSV file
    # — this is the basis for later analysis in TensorBoard
    env = Monitor(env, filename=os.path.join(CONFIG['log_dir'], 'monitor'))
    return env

def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)

    print("Creating environment...")
    env = make_env()

    print("Creating SAC model...")

    ###### WHEN CREATING NEW MODEL ######

    hp = sac_hyperparams()
    model = SAC(
        policy='MlpPolicy',
        env=env,
        tensorboard_log=CONFIG['log_dir'],
        verbose=1,
        **hp,
    )

    ###### WHEN LOADING EXISTING MODEL ######

    #model = SAC.load('models/ ##path## ', env=env)
    apply_finetune_overrides(model)

    # ================================================================
    # CALLBACKS — what to do during training
    # ================================================================


    #Creating Replay Buffer

    # if TRAIN_MODE == 'finetune' and CONFIG.get('prefill_expert_buffer'):
    #     import pickle

    #     path = CONFIG['expert_data_path']
    #     print(f"Loading expert data to replay buffer: {path}")
    #     with open(path, 'rb') as f:
    #         expert_data = pickle.load(f)

    #     # Rewards in pkl = old function; critic gets new ones from env in learn() anyway.
    #     # After changing reward, worth collecting expert_data again.
    #     for i in range(len(expert_data) - 1):
    #         row = expert_data[i]
    #         model.replay_buffer.add(
    #             row['obs'],
    #             expert_data[i + 1]['obs'],
    #             row['action'],
    #             row.get('reward', 0.0),
    #             row.get('done', False),
    #             [{}],
    #         )
    #     print(f"Buffer after prefill: {model.replay_buffer.size()} transitions")


    # Saves model every save_freq steps
    checkpoint_cb = CheckpointCallback(
        save_freq=CONFIG['save_freq'],
        save_path=CONFIG['save_dir'],
        name_prefix='sac_torcs_14',
        verbose=1,
    )

    hp = sac_hyperparams()
    print(f"Mode: {TRAIN_MODE} | lr={hp['learning_rate']} | buffer={hp['buffer_size']}")
    print(f"ent_coef={hp['ent_coef']} | learning_starts={hp['learning_starts']}")
    print(f"Starting training — {CONFIG['total_timesteps']:,} steps")
    print(f"Learning every {CONFIG['train_freq']} step(s), gradient_steps={hp['gradient_steps']}")
    print("=" * 50)

    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(CONFIG['save_dir'], 'sac_torcs_final1')
    model.save(final_path)
    print(f"\nModel saved: {final_path}")

    env.close()

if __name__ == '__main__':
    main()