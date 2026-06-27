# 🏎️ IBM AI Racing League - Team FerrAIri

<div align="center">

**AGH University of Science and Technology | AI LAB**

*Combining Expert Knowledge with Deep Reinforcement Learning for Superhuman Racing Performance*

</div>

---

## Team Members
- **Fabian Bicca** 
- **Józef Kasprzycki**
- **Mikołaj Klima**
- **Konrad Leżoń**

---

## Project Overview

We developed an AI racing agent for the TORCS simulator that achieves **superhuman performance** on the challenging Corkscrew track through a novel three-stage training pipeline. Our approach combines the stability of rule-based systems with the adaptability of deep reinforcement learning, resulting in an agent that not only matches but **exceeds expert human performance**.

### Key Achievements
- **Expert-Level Performance** - Fast and precise steering control
- **Smooth Driving** - Anti-slalom reward shaping reduces oscillations
- **Robust Architecture** - Created for handling track variations and edge cases
- **Efficient Learning** - Behavioral Cloning improves the agent's performance

---

## Results

Rule-Based Model:
   - 102s

Reinforcement Learning Model:
   - 110s

---

## Methodology

Our three-stage approach leverages the strengths of both classical and modern AI techniques:

### Stage 1: Rule-Based Expert System
We developed a sophisticated parametrized driver based on racing heuristics:
- **Track Analysis** - Sector-based speed optimization
- **Adaptive Control** - Dynamic steering and throttle adjustment
- **Traction Control** - Prevents wheel slip on acceleration
- **Automated Tuning** - Hyperparameter search optimizes 20+ parameters

**Key Innovation:** Our expert system uses track sector analysis to predict optimal racing lines and adjust behavior for different track segments (straights, turns, chicanes).

### Stage 2: Behavioral Cloning
Pre-training through imitation learning provides a strong initialization:
- **Expert Data Collection** - 40+ laps of optimal driving (~50K transitions)
- **MLP Architecture** - 256-256 hidden layers with Tanh activation
- **Supervised Learning** - MSE loss on expert actions
- **Weight Transfer** - Direct initialization of SAC actor network

**Key Innovation:** BC eliminates the dangerous random exploration phase, allowing the agent to start with safe, competitive driving behavior.

### Stage 3: Reinforcement Learning Fine-Tuning
SAC (Soft Actor-Critic) pushes beyond expert performance:
- **Continuous Control** - 2D action space (steering, throttle/brake)
- **Rich Observations** - 30D state space (speeds, track sensors, wheel spin)
- **Shaped Rewards** - Multi-component reward balancing speed, safety, and smoothness
- **Entropy Regularization** - Carefully tuned to prevent slalom behavior

**Key Innovation:** Our custom reward function includes anti-slalom penalties that weight steering smoothness based on track geometry, preventing the oscillatory behavior common in RL racing agents.

---

## Architecture

### Observation Space (30 dimensions)
```
- Speed Vector (3D): speedX, speedY, speedZ
- Track Position (1D): lateral position [-1, 1]
- Angle (1D): car orientation relative to track
- Track Sensors (19D): distance to track edges
- Wheel Spin (4D): individual wheel velocities
- Engine State (2D): RPM, current gear
```

### Action Space (2 dimensions)
```
- Steering: continuous [-1, 1] (left to right)
- Pace: continuous [-1, 1] (brake to throttle)
```

### Neural Network Architecture
```
Input (30) → Dense(256) → ReLU → Dense(256) → ReLU → Output(2) → Tanh
```

### Reward Function Components
Our carefully crafted reward function balances multiple objectives:

1. **Speed Reward** (+) - `speed × cos(angle) / 300`
   - Encourages forward velocity aligned with track
   
2. **Progress Reward** (+) - `Δ(distFromStart) / 2`
   - Rewards advancement along track
   
3. **Position Penalty** (-) - `0.3 × |trackPos|`
   - Keeps car near racing line
   
4. **Angle Penalty** (-) - `0.1 × |angle/π|`
   - Maintains proper orientation
   
5. **Lateral Speed Penalty** (-) - `0.4 × |speedY/50|`
   - Reduces sliding and drifting
   
6. **Position Change Penalty** (-) - `0.8 × |Δ(trackPos)|`
   - **Anti-slalom component** - prevents oscillation
   
7. **Steering Smoothness Penalty** (-) - `weight × |Δ(steer)|`
   - Dynamic weight based on track geometry
   - Higher penalty on straights, lower in turns
   
8. **Steering-Throttle Penalty** (-) - `0.2 × |steer × throttle|`
   - Discourages aggressive steering at high speed
   
9. **Damage Penalty** (-) - `0.1 × Δ(damage)`
   - Heavily penalizes collisions

---

## Project Structure

```
gym_torcs/
├── Expert System
│   ├── expert_driver.py          # Runs the expert driver
│   ├── expert_env.py             # Expert wrapper for data collection
│   ├── autoresearch_best.json    # Optimized hyperparameters
│   └── track_sectors.py          # Track geometry analysis
│
├── Data Collection
│   ├── collect_expert_data.py    # Automated expert data gathering
│   └── expert_data.pkl           # Collected expert trajectories
│
├── Training Pipeline
│   ├── behavioral_cloning.py     # BC pre-training implementation
│   ├── train.py                  # SAC fine-tuning with callbacks
│   ├── torcs_env.py              # Gym-compatible TORCS environment
│   └── snakeoil.py               # TORCS client communication
│
├── Evaluation
│   └── evaluate.py               # Model testing
│
├── Models & Logs
│   ├── models/                   # Saved model checkpoints
│   ├── logs/                     # TensorBoard training logs
│   └── bc_model.pth              # Pre-trained BC model
│
└── Configuration
    ├── requirements.txt          # Python dependencies
    ├── practice.xml              # TORCS race configuration
    └── README.md                 # This file
```

---

## Quick Start

### Prerequisites
- **Python:** 3.10 or higher
- **GPU:** CUDA-compatible GPU recommended (training)
- **TORCS:** Modified version included in`vtorcs-RL-color`             installed using skillsbuild tutorial

### Installation

1. **Install TORCS**
   **Download the modified version of TORCS with skillsbuild tutorial**

2. **Clone the repository**
```bash
git clone <repository-url>
cd gym_torcs
```

3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure TORCS**
   Launch TORCS 
   
   In GUI: Race → Practice → Configure Race
   Select track: Corkscrew
   Select car: Any GT car
   Race → Practice → New Race (opens server)
```

---

## Usage

### Training from Scratch

**Step 1: Collect Expert Data**
```bash
python collect_expert_data.py
```
This runs the optimized expert driver for 40 laps and saves trajectories to `expert_data.pkl`.

**Step 2: Behavioral Cloning Pre-training**
```bash
python behavioral_cloning.py
```
Trains a neural network to imitate expert behavior. Saves model to `bc_model.pth`.

**Step 3: RL Fine-tuning**
```bash
python train.py
```
Fine-tunes the BC-initialized agent using SAC. Models saved every 100K steps to `models/`.

### Training Configuration

Edit `train.py` to customize training:

```python
CONFIG = {
    'port': 3001,                    # TORCS server port
    'total_timesteps': 1_000_000,    # Training duration
    'learning_starts': 10_000,       # Random exploration steps
    'batch_size': 256,               # Replay buffer batch size
    'gamma': 0.99,                   # Discount factor
    'tau': 0.005,                    # Target network update rate
    'gradient_steps': 1,             # Updates per environment step
    'save_freq': 100_000,            # Checkpoint frequency
}
```

**Training Modes:**
- `TRAIN_MODE = 'scratch'` - Train from random initialization
- `TRAIN_MODE = 'finetune'` - Continue from BC or checkpoint

### Evaluation

```bash
python evaluate.py
```

Runs the trained agent for multiple laps and reports:
- Average lap time
- Best lap time
- Success rate (completed laps)
- Track position statistics

---

## 📊 Monitoring Training

We use TensorBoard for real-time training visualization:

```bash
tensorboard --logdir=logs/
```

**Key Metrics to Monitor:**
- `rollout/ep_rew_mean` - Average episode reward (should increase)
- `rollout/ep_len_mean` - Episode length (longer = fewer crashes)
- `train/actor_loss` - Policy network loss
- `train/critic_loss` - Value network loss
- `train/ent_coef` - Entropy coefficient (exploration vs exploitation)

---

## 🛠️ Technical Details

### Hyperparameter Tuning

Our final hyperparameters were selected through extensive experimentation:

| Parameter | Scratch | Finetune | Rationale |
|-----------|---------|----------|-----------|
| Learning Rate | 3e-4 | 1e-4 | Lower LR for fine-tuning prevents catastrophic forgetting |
| Entropy Coef | auto | 0.05 | Fixed low entropy prevents slalom in fine-tuning |
| Buffer Size | 300K | 300K | Balances memory usage and sample diversity |
| Gradient Steps | 1 | 2 | More updates per step improves sample efficiency |
| Learning Starts | 10K | 0 | BC initialization eliminates need for random exploration |

### Anti-Slalom Innovation

Traditional RL racing agents often develop oscillatory "slalom" behavior due to:
1. High entropy encouraging exploration
2. Reward functions that don't penalize lateral movement
3. Lack of smoothness constraints

**Our Solution:**
- **Dynamic Steering Penalty** - Weight increases on straights, decreases in turns
- **Track Geometry Awareness** - Uses forward-looking sensors (7-11) to detect straight sections
- **Position Change Penalty** - Directly penalizes lateral oscillation
- **Low Entropy Fine-tuning** - Reduces random exploration after BC initialization

### Automatic Gear Shifting

We implement rule-based gear shifting to simplify the learning problem:

```python
def auto_gear(speed, current_gear):
    up_thresholds = [60, 100, 140, 180, 220]    # km/h
    down_thresholds = [40, 80, 120, 160, 200]   # km/h with hysteresis
    
    if current_gear < 6 and speed > up_thresholds[current_gear - 1]:
        return current_gear + 1
    elif current_gear > 1 and speed < down_thresholds[current_gear - 2]:
        return current_gear - 1
    return current_gear
```

This reduces action space dimensionality and prevents gear-related failures.

---

## 🔧 Troubleshooting

### Common Issues

**TORCS won't connect**
```bash
# Check if TORCS is running
ps aux | grep torcs

# Kill existing instances
pkill torcs

# Restart with correct port
torcs -p 3001
```

**Training crashes immediately**
- Ensure TORCS server is running before starting training
- Check that port in `train.py` matches TORCS configuration
- Verify GPU memory availability (reduce batch_size if needed)

**Agent drives off track repeatedly**
- Increase `learning_starts` for more random exploration
- Check reward function weights in `torcs_env.py`
- Verify BC model was loaded correctly

**Slalom behavior persists**
- Reduce `ent_coef` in fine-tuning mode
- Increase `penalty_pos_delta` weight in reward function
- Ensure `_straight_steer_weight()` is functioning correctly

**Memory leak / TORCS slowdown**
- TORCS has known memory leak on episode reset
- Restart TORCS every 50-100 episodes
- Monitor system memory usage

---

## 🤖 IBM Tools Integration

### IBM Granite
We leveraged IBM Granite models for:
- **Research & Literature Review** - Summarizing RL papers and racing strategies
- **Code Documentation** - Generating comprehensive docstrings
- **Hyperparameter Suggestions** - Initial parameter ranges for tuning
- **Debugging Assistance** - Analyzing training curves and suggesting fixes

### IBM SkillsBuild
Team members completed courses on:

![Skillsbuild Badges](<Skillsbuild.png>)

These courses provided foundational knowledge that directly informed our architectural decisions and training strategies.

### IBM Bob (AI Assistant)
Bob was instrumental throughout development:
- **Code Development** - Assisted in implementing complex reward functions
- **Debugging** - Helped identify and fix training instabilities
- **Documentation** - Helped to structure the repository

**Example Use Case:** When our agent developed slalom behavior, Bob helped us:
1. Analyze the reward function for missing penalties
2. Implement the dynamic steering weight based on track geometry
3. Add position change penalty to directly discourage oscillation
4. Tune entropy coefficient for stable fine-tuning

---

## 📚 References & Inspiration

### Academic Papers
- **SAC Algorithm:** [Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL](https://arxiv.org/abs/1801.01290)
- **TORCS for AI:** [The TORCS Racing Board Game](https://arxiv.org/abs/1304.1672)
- **Behavioral Cloning:** [A Reduction of Imitation Learning and Structured Prediction](https://arxiv.org/abs/1011.0686)

### Open Source Projects
- **Stable-Baselines3:** High-quality RL implementations
- **Gym-TORCS:** Original TORCS-Gym wrapper
- **vtorcs:** Modified TORCS for RL research

### Racing Strategy
- Real-world racing techniques (racing line, trail braking, throttle control)
- Professional sim-racing best practices
- Track-specific optimization for Corkscrew circuit

---

## 🎓 Key Learnings

1. **Behavioral Cloning is Crucial** - BC initialization improved the agent's performance and eliminated random exploration phase

2. **Reward Shaping Matters** - Carefully designed multi-component rewards with anti-slalom penalties were essential for smooth driving

3. **Entropy Control** - High entropy during scratch training, low entropy during fine-tuning prevents policy degradation

4. **Domain Knowledge Helps** - Incorporating racing heuristics (gear shifting, traction control) simplified the learning problem

5. **Monitoring is Essential** - TensorBoard visualization helped us quickly identify and fix training issues

---

## 🔮 Future Improvements

- [ ] Multi-track generalization
- [ ] Opponent awareness and overtaking
- [ ] Weather and track condition adaptation
- [ ] Real-time strategy adjustment

---

## 📄 License

This project is developed for the IBM AI Racing League competition.

---

## 🙏 Acknowledgments

- **IBM** for organizing the AI Racing League and providing tools
- **AGH University AI LAB** for resources and support
- **TORCS Community** for the simulator and documentation
- **Stable-Baselines3 Team** for excellent RL implementations

---

<div align="center">

**Team FerrAIri** | AGH University of Science and Technology

*Racing towards the future of AI* 🏁

</div>
