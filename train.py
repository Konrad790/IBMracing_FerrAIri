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
# KONFIGURACJA — tu zmieniasz parametry nie ruszając reszty kodu
# ================================================================
CONFIG = {
    # Środowisko
    'port': 3001,

    # Trening
    'total_timesteps': 500_000,   # ile kroków łącznie
    'learning_starts': 10_000,    # ile kroków zebrać zanim zaczniesz uczyć
                                  # (żeby replay buffer nie był pusty)

    # SAC — hiperparametry
    'learning_rate': 3e-4,        # jak duże kroki robi optymalizator
    'buffer_size': 100_000,       # ile doświadczeń trzymać w pamięci
    'batch_size': 256,            # ile losować z bufora na raz
    'gamma': 0.99,                # współczynnik dyskontowania przyszłych nagród
    'tau': 0.005,                 # jak szybko target network śledzi główną sieć
    'train_freq': 1,              # co ile kroków aktualizować sieć
    'gradient_steps': 1,          # ile aktualizacji na raz

    # Zapis
    'save_dir': 'models',
    'log_dir': 'logs',
    'save_freq': 50_000,          # co ile kroków zapisywać checkpoint
}

def make_env():
    """Tworzy i opakuje środowisko w Monitor (loguje epizody)."""
    env = TorcsEnv(port=CONFIG['port'])

    # Monitor zapisuje nagrody i długości epizodów do pliku CSV
    # — to podstawa do późniejszej analizy w TensorBoard
    env = Monitor(env, filename=os.path.join(CONFIG['log_dir'], 'monitor'))
    return env

def main():
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['log_dir'], exist_ok=True)

    print("Tworzenie środowiska...")
    env = make_env()

    print("Tworzenie modelu SAC...")

    ###### JAK TWORZYCIE NOWY MODEL ######

    # model = SAC(
    #     policy='MlpPolicy',       # MLP = zwykła sieć feed-forward
    #                               # (alternatywa: CnnPolicy dla obrazów)
    #     env=env,

    #     # Hiperparametry z konfiguracji
    #     learning_rate=CONFIG['learning_rate'],
    #     buffer_size=CONFIG['buffer_size'],
    #     batch_size=CONFIG['batch_size'],
    #     gamma=CONFIG['gamma'],
    #     tau=CONFIG['tau'],
    #     train_freq=CONFIG['train_freq'],
    #     gradient_steps=CONFIG['gradient_steps'],
    #     learning_starts=CONFIG['learning_starts'],

    #     # Architektura sieci — dwie warstwy po 256 neuronów
    #     # dla aktora i krytyka
    #     policy_kwargs=dict(net_arch=[256, 256]),

    #     # Logowanie do TensorBoard
    #     tensorboard_log=CONFIG['log_dir'],

    #     verbose=1,   # wypisuj postęp w terminalu
    # )

    ###### JAK WCZYTUJECIE JAKIS ISTNIEJACY ######

    model = SAC.load('models/sac_new_v3_torcs_100000_steps', env=env)

    # ================================================================
    # CALLBACKS — co robić w trakcie treningu
    # ================================================================

    # Zapisuje model co save_freq kroków
    checkpoint_cb = CheckpointCallback(
        save_freq=CONFIG['save_freq'],
        save_path=CONFIG['save_dir'],
        name_prefix='sac_new_v4_torcs',
        verbose=1,
    )

    print(f"Rozpoczynam trening — {CONFIG['total_timesteps']:,} kroków")
    print(f"Pierwsze {CONFIG['learning_starts']:,} kroków: zbieranie doświadczeń (losowe akcje)")
    print(f"Potem: uczenie co {CONFIG['train_freq']} krok(ów)")
    print("=" * 50)

    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=checkpoint_cb,
        progress_bar=True,
    )

    # Zapisz finalny model
    final_path = os.path.join(CONFIG['save_dir'], 'sac_torcs_final')
    model.save(final_path)
    print(f"\nModel zapisany: {final_path}")

    env.close()

if __name__ == '__main__':
    main()