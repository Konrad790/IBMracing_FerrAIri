import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import SAC
from torcs_env import TorcsEnv

def train_bc(data_path='expert_data.pkl', epochs=50, batch_size=256, lr=3e-4):
    """Phase 1 — Behavioral Cloning."""
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Expert data: {len(data)} steps")

    # Prepare data
    obs     = torch.FloatTensor(np.array([d['obs']    for d in data]))
    actions = torch.FloatTensor(np.array([d['action'] for d in data]))

    dataset    = TensorDataset(obs, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Simple MLP network — same architecture as SAC actor
    model = nn.Sequential(
        nn.Linear(obs.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, actions.shape[1]),
        nn.Tanh()  # output in [-1, 1]
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    print("Training Behavioral Cloning...")
    for epoch in range(epochs):
        total_loss = 0
        for obs_batch, action_batch in dataloader:
            optimizer.zero_grad()
            predicted = model(obs_batch)
            loss      = loss_fn(predicted, action_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), 'bc_model.pth')
    print("BC model saved: bc_model.pth")
    return model

def transfer_bc_to_sac(bc_model, sac_model):
    """
    Phase 2 — transfer BC weights to SAC actor.
    BC and SAC have the same MLP 256-256 architecture.
    """
    # Get weights from BC
    bc_state = bc_model.state_dict()
    
    # Get SAC actor weights
    sac_actor_state = sac_model.policy.actor.state_dict()
    
    # BC → SAC actor layer mapping
    # SAC uses names: latent_pi.0, latent_pi.2, mu
    mapping = {
        '0.weight': 'latent_pi.0.weight',
        '0.bias':   'latent_pi.0.bias',
        '2.weight': 'latent_pi.2.weight',
        '2.bias':   'latent_pi.2.bias',
        '4.weight': 'mu.weight',
        '4.bias':   'mu.bias',
    }

    for bc_key, sac_key in mapping.items():
        if bc_key in bc_state and sac_key in sac_actor_state:
            sac_actor_state[sac_key] = bc_state[bc_key]
            print(f"Transferred: {bc_key} → {sac_key}")

    sac_model.policy.actor.load_state_dict(sac_actor_state)
    print("BC weights transferred to SAC actor")
    return sac_model

if __name__ == '__main__':
    # Phase 1 — Behavioral Cloning
    bc_model = train_bc(
        data_path='expert_data.pkl',
        epochs=100,
        batch_size=256,
    )

    # Phase 2 — transfer weights to SAC and fine-tune with RL
    print("\nCreating SAC environment...")
    env = TorcsEnv(port=3001)
    
    sac_model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        learning_starts=1000,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
    )

    # Transfer BC weights to SAC
    sac_model = transfer_bc_to_sac(bc_model, sac_model)

    # # Fine-tune with RL
    # print("\nPhase 2 — RL fine-tuning...")
    # sac_model.learn(total_timesteps=500_000)
    # sac_model.save('models/sac_bc_finetuned')

    sac_model.save('models/sac_bc_pretrained')
    print("Model saved: models/sac_bc_pretrained")
    
    env.close()