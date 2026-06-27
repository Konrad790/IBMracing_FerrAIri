import numpy as np
import time
from stable_baselines3 import SAC
from torcs_env import TorcsEnv

def wait_for_torcs(env, timeout_s=60, retry_interval_s=2):
    """
    Czeka aż TORCS zacznie odpowiadać na porcie UDP.
    Zwraca pierwszą obserwację i info po udanym reset().
    """
    start_time = time.time()
    attempt = 1
    last_error = None

    while time.time() - start_time < timeout_s:
        try:
            obs, info = env.reset()
            print(f"TORCS gotowy (proba {attempt}).")
            return obs, info
        except Exception as err:
            last_error = err
            elapsed = int(time.time() - start_time)
            print(
                f"[{elapsed:02d}s] TORCS jeszcze niegotowy "
                f"(proba {attempt}): {err}"
            )
            time.sleep(retry_interval_s)
            attempt += 1

    raise RuntimeError(
        f"Timeout po {timeout_s}s: TORCS nadal nie odpowiada na porcie {env.port}.\n"
        "Uruchom TORCS, wejdz do wyscigu SCR i sprobuj ponownie."
    ) from last_error

def evaluate(model_path, episodes=5):
    env = TorcsEnv(port=3001)
    model = SAC.load(model_path, env=env)
    
    print(f"Loaded model: {model_path}")
    print("Czekam na gotowosc TORCS (bez wciskania Enter)...")

    for episode in range(episodes):
        obs, _ = wait_for_torcs(env, timeout_s=90, retry_interval_s=2)
        total_reward = 0
        steps = 0

    prev_dist = None
    
    while True:
        # deterministic=True = agent nie eksploruje, tylko eksploatuje
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        dist = info['distFromStart']
        if prev_dist is None:
            progress = 0.0
        else:
            progress = max(0.0, dist - prev_dist)
        prev_dist = dist

        if steps % 20 == 0:
            print(f"Progress: {progress:.1f} | distance: {dist:.1f}")

        if terminated or truncated:
            print(f"Epizod {episode+1} | nagroda: {total_reward:.1f} | kroki: {steps}")
            break

    env.close()

if __name__ == '__main__':
    evaluate('models/sac_torcs_8_300000_steps')
