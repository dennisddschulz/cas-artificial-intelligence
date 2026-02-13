# DQN vs Double DQN Hyperparameter Analysis - Complete Documentation

## 📋 Project Overview

This comprehensive analysis examines how hyperparameter choices affect the performance and training dynamics of DQN and Double DQN algorithms on the Lunar Lander-v3 environment.

## 🎯 Main Objectives

1. **Understand Learning Dynamics**: How different hyperparameters affect learning speed, stability, and final performance
2. **Compare Algorithms**: Identify when and why Double DQN outperforms DQN
3. **Provide Practical Recommendations**: Evidence-based hyperparameter choices for this environment
4. **Analyze Trade-offs**: Performance vs. training time, stability vs. exploration

## 🔬 Experimental Setup

### Environment: Lunar Lander-v3

**Task**: Safely land a spacecraft on the moon

**State Space** (8 continuous features):
- Position: x, y (normalized coordinates)
- Velocity: vx, vy (derivatives of position)
- Angle: spacecraft rotation
- Angular velocity: rate of rotation
- Left leg contact flag (binary)
- Right leg contact flag (binary)

**Action Space** (4 discrete actions):
- 0: Do nothing
- 1: Fire left engine
- 2: Fire main engine
- 3: Fire right engine

**Reward Signal**:
- -1 per timestep (encourages quick landing)
- +100 for successful landing
- -100 for crash
- Total episode reward target: > 200

**Episode Length**: Typically 200-500 timesteps

### Baseline Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Network Architecture | MLP with 1 hidden layer | Simple, sufficient for state representation |
| Hidden Units | 128 | Balance between capacity and training speed |
| Learning Rate | 1×10⁻³ | Standard for control tasks |
| Discount Factor (γ) | 0.99 | Long-term reward consideration |
| Batch Size | 64 | Good balance between gradient quality and stability |
| Replay Buffer | 100,000 experiences | Sufficient diversity without excessive memory |
| Min Buffer | 1,000 experiences | Warmup period before learning |
| ε-start | 1.0 | Full exploration initially |
| ε-end | 0.01 | Minimal exploration at end |
| Decay Schedule | Linear over 250k steps | Gradual transition |
| Target Update | Hard every 1,000 steps | Stable value estimates |
| Gradient Clipping | Enabled | Prevents divergence |

## 📊 Experimental Design

### Experiment 1: Replay Buffer Size Analysis

**Research Question**: How does buffer size affect learning speed and stability?

**Variants Tested**:
- 10,000 experiences (small)
- 50,000 experiences (medium-small)
- 100,000 experiences (medium)
- 200,000 experiences (large)

**Hypotheses**:
- Small buffers: Quick learning but high variance (limited diversity)
- Large buffers: Better gradient estimates but slower updates
- Optimal exists: Balance between diversity and update frequency

**Metrics**:
- Episode return curves
- Final performance (mean ± std of last 100 episodes)
- Learning stability (standard deviation)
- Convergence speed

---

### Experiment 2: Exploration Strategy (ε-decay) Analysis

**Research Question**: How does exploration scheduling affect convergence?

**Variants Tested**:
- Decay over 50,000 steps (fast exploration → fast exploitation)
- Decay over 150,000 steps (moderate)
- Decay over 250,000 steps (slow, baseline)
- Decay over 400,000 steps (very slow)

**Hypotheses**:
- Fast decay: Quick initial improvement but poor final performance (premature convergence)
- Slow decay: Better exploration, more robust but potentially slower initial learning
- Optimal decay depends on environment complexity

**Metrics**:
- Learning curves at different decay speeds
- Final performance comparison
- Early learning speed (episodes to reach score 0, 50, 100, etc.)
- Stability of convergence

---

### Experiment 3: Target Network Update Strategy

**Research Question**: How should target network weights be updated?

**Update Methods**:

1. **Hard Updates** (θ⁻ ← θ every N steps):
   - Complete replacement every N training steps
   - Variants: 500, 1,000, 2,000 steps
   - Pro: Explicit control over update frequency
   - Con: Sudden shifts in target values

2. **Soft Updates** (θ⁻ ← τθ + (1-τ)θ⁻ at every step):
   - Polyak averaging
   - Variants: τ = 0.001, 0.005, 0.01, 0.05
   - Pro: Smooth, continuous target evolution
   - Con: Requires careful τ tuning

**Hypotheses**:
- Hard updates: More stable but discrete value jumps
- Soft updates: Smoother learning, less variance
- Interaction with buffer size and learning rate

**Metrics**:
- Convergence smoothness
- Final performance variance
- Learning curve shape

---

### Experiment 4: Learning Parameters (Learning Rate & Batch Size)

**Research Question**: How sensitive is the algorithm to learning rate and batch size?

**Learning Rates**:
- 0.5×10⁻³ (conservative)
- 1.0×10⁻³ (baseline)
- 2.0×10⁻³ (aggressive)

**Batch Sizes**:
- 32 (small, noisier gradients)
- 64 (baseline)
- 128 (large, smoother gradients)

**Hypotheses**:
- Higher LR: Faster learning but instability risk
- Lower LR: Stable but slow
- Larger batches: Better gradient estimates but slower updates
- Interaction effects between LR and batch size

**Metrics**:
- Convergence speed
- Final performance
- Training stability (variance)
- Gradient variance during training

---

## 🧠 Algorithm Comparison: DQN vs Double DQN

### The Overestimation Problem

In standard DQN, the target is computed as:
```
y = r + γ·max_a' Q(s', a')  [using target network]
```

This can lead to **Q-value overestimation** because:
1. The same network selects AND evaluates actions
2. Systematic bias towards overestimating high values
3. Causes poor policy (exploits overestimated actions)

### Double DQN Solution

Double DQN decouples selection and evaluation:
```
a* = argmax_a' Q_online(s', a')      [selection with online network]
y = r + γ·Q_target(s', a*)            [evaluation with target network]
```

Benefits:
- Reduces overestimation bias
- More stable learning
- Better final performance in many domains

### When Does Double DQN Help Most?

1. **With poor hyperparameters**: More robust to suboptimal choices
2. **Early in training**: Better value estimates prevent bad behaviors
3. **With large action spaces**: More overestimation to correct
4. **In complex environments**: More value updates → more bias accumulation

### When Is DQN Sufficient?

1. With well-tuned hyperparameters
2. In simple environments (small action space)
3. When overestimation isn't a major factor

---

## 📈 Expected Results & Interpretations

### Experiment 1: Buffer Size Impact

**Expected Findings**:
- **10k buffer**: 
  - Fast initial learning
  - High variance (noisy gradient estimates)
  - Lower final performance
  - Instability in late training

- **50-100k buffer**:
  - Best balance
  - Smooth learning curves
  - Stable convergence
  - Good final performance

- **200k buffer**:
  - Similar final performance to 100k
  - Slightly smoother curves
  - Higher memory requirements (diminishing returns)

**Interpretation**:
Larger buffers provide better diversity of experiences, leading to more stable gradient estimates. However, beyond 100k, improvements are marginal.

---

### Experiment 2: Epsilon Decay Impact

**Expected Findings**:
- **50k decay (fast)**:
  - Quick initial improvement
  - Early performance drop
  - Suboptimal final policy (premature exploitation)

- **150-250k decay**:
  - Balanced exploration-exploitation
  - Smooth learning curves
  - Good final performance

- **400k decay (slow)**:
  - Longer exploration phase
  - Excellent final performance
  - Slower convergence (more episodes needed)

**Interpretation**:
Too-fast exploration decay causes the agent to commit to suboptimal policies before learning good strategies. Too-slow decay wastes episodes on exploration. Optimal is problem-dependent.

---

### Experiment 3: Update Strategy Impact

**Expected Findings**:
- **Hard updates every 500 steps**:
  - More stable updates
  - Good convergence
  - Still within reasonable range

- **Hard updates every 1000 steps**:
  - Optimal balance (baseline)
  - Stable learning
  - Responsive target network

- **Hard updates every 2000 steps**:
  - Target network becomes stale
  - Potential divergence
  - Delayed learning

- **Soft updates with τ=0.005**:
  - Very smooth learning
  - Lower variance
  - Slightly slower convergence

- **Soft updates with τ=0.001**:
  - Too slow, target network barely changes
  - Behaves similarly to infrequent hard updates

**Interpretation**:
Hard updates every 1000 steps provide good stability. Soft updates work well but require careful τ tuning (0.005 is better than 0.001).

---

### Experiment 4: Learning Parameters Impact

**Expected Findings**:
- **Learning Rate 0.5×10⁻³**:
  - Very stable but slow
  - Eventually converges
  - Conservative approach

- **Learning Rate 1.0×10⁻³**:
  - Optimal: good speed and stability
  - Baseline

- **Learning Rate 2.0×10⁻³**:
  - Faster initial learning
  - Potential instability in late training
  - Higher variance

- **Batch Size 32**:
  - Faster updates
  - Higher gradient variance
  - Noisier learning curves

- **Batch Size 64**:
  - Balance between speed and stability
  - Baseline (good)

- **Batch Size 128**:
  - Better gradient estimates
  - Slower update frequency
  - Smoother but potentially slower

**Interpretation**:
1e-3 and batch size 64 work well for Lunar Lander.

---

## 🎯 Key Insights

### Insight 1: Buffer Size vs. Learning Quality

Larger buffers improve gradient estimates but with diminishing returns. 100k is a sweet spot for Lunar Lander.

### Insight 2: Exploration Timing Matters

The exploration schedule is critical. Decaying too fast causes the agent to lock into suboptimal strategies. Giving enough time for exploration is essential.

### Insight 3: Double DQN's Robustness

Double DQN shows more consistent performance across different hyperparameter settings. This makes it more reliable in practice where optimal hyperparameters may not be known a priori.

### Insight 4: Update Frequency is Crucial

Target network update frequency should be tuned carefully. Too frequent = wasted computation, too infrequent = stale targets.

### Insight 5: Learning Rate and Batch Size Interact

These parameters aren't independent. Lower learning rates can use larger batches, and vice versa.

---

## 📊 How to Interpret the Plots

### Plot 1: Buffer Size Analysis
- **Left column (DQN)**: Shows how buffer size affects DQN learning
- **Right column (Double DQN)**: Shows corresponding Double DQN results
- **Top row**: Episode return curves with moving average
- **Bottom left**: Final performance bars (higher is better)
- **Bottom right**: Stability (lower std = more consistent)

### Plot 2: Epsilon Decay Analysis
- Shows effect of decay speed on learning curves
- Fast decay leads to earlier convergence but worse final performance
- Slow decay explores longer, achieving better final results
- Double DQN more robust to decay speed variations

### Plot 3: Update Strategy
- Compares hard updates (different frequencies) vs. soft updates (different τ)
- Hard updates every 1000 steps is sweet spot
- Soft updates with τ=0.005 comparable or slightly better

### Plot 4: Learning Parameters
- Top left: Learning rate effect on DQN
- Top right: Batch size effect on DQN
- Bottom: Final performance comparison for both parameters
- Shows optimal LR=1e-3, batch=64

---

## 🛠️ Implementation Details

### DQN Algorithm

```python
def train_dqn_episode():
    state = env.reset()
    for step in range(max_steps):
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        
        if buffer.ready():
            batch = buffer.sample()
            
            # Current Q-values
            q = network(state_batch)[actions]
            
            # Target Q-values (using target network for stability)
            next_q_values = target_network(next_state_batch)
            target = reward_batch + gamma * next_q_values.max(dim=1)[0]
            
            # Loss and update
            loss = MSE(q, target)
            optimizer.step(loss)
        
        if training_step % update_freq == 0:
            hard_update(target_network, network)
        
        state = next_state
```

### Double DQN Modification

```python
# Only change in replay() method:
if use_double_dqn:
    # Select action with online network
    next_actions = network(next_state_batch).argmax(dim=1)
    # Evaluate with target network
    next_q_values = target_network(next_state_batch).gather(1, next_actions)
else:
    # Standard DQN: both with target network
    next_q_values = target_network(next_state_batch).max(dim=1)[0]
```

---

## 🔍 Analysis Methodology

1. **Reproducibility**: Fixed random seeds for all experiments
2. **Multiple Runs**: Each configuration run multiple times (seeds 0-4)
3. **Smoothing**: Moving average windows (100-200 episodes) to reduce noise
4. **Statistics**: Mean, std, min, max reported for final 100 episodes
5. **Visualization**: Professional plots with clear legends and labels

---

## 💡 Practical Recommendations

### For Production Systems
1. Use **Double DQN** for robustness
2. Set buffer to **100,000** (good balance)
3. Decay ε over **250,000 steps** (linear)
4. Learning rate: **1×10⁻³**
5. Batch size: **64**
6. Hard update target every **1,000 steps**

### For Limited Resources
1. Use **DQN** (faster training)
2. Buffer: **50,000** (reduce memory)
3. Faster decay: **150,000 steps**
4. Increase learning rate: **2×10⁻³** (for faster convergence)

### For Research/Exploration
1. Try **both** DQN and Double DQN
2. Sweep epsilon decay schedules
3. Compare hard vs. soft updates
4. Analyze learning rate sensitivity

---

## 📁 Generated Files

```
results/
├── 01_buffer_size_analysis.png           # Experiment 1 results
├── 02_epsilon_decay_analysis.png         # Experiment 2 results
├── 03_update_strategy_analysis.png       # Experiment 3 results
├── 04_learning_rate_batch_size_analysis.png  # Experiment 4 results
├── ANALYSIS_REPORT.txt                   # Detailed statistics
└── DQN_vs_DoubleDQN_Analysis.pptx       # Presentation slides
```

---

## ⏱️ Runtime Estimates

| Configuration | Episodes | Time (GPU) | Time (CPU) |
|---------------|----------|-----------|-----------|
| Quick test | 100 | 15-30 min | 1-2 hours |
| Standard | 300 | 45-90 min | 3-5 hours |
| Full analysis | 500+ | 2-4 hours | 6-12 hours |

Current setup uses 300 episodes per configuration for balance between thoroughness and runtime.

---

## 🚀 How to Run the Analysis

```bash
# Navigate to directory
cd /home/isc-den/cas-artificial-intelligence/10_dqn/lunar_lander_new

# Run hyperparameter analysis
python dqn_hyperparameter_analysis.py

# Create presentation slides
python create_presentation.py

# View results
ls -la results/
```

---

## 📚 References

**Key Papers**:
- DQN: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- Double DQN: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
- Dueling Networks & Prioritized Experience Replay: Further improvements

**Environment**:
- Lunar Lander-v3: Part of Gymnasium (formerly OpenAI Gym)
- Classic control environment from DeepMind

---

## ❓ FAQ

**Q: Why does Double DQN sometimes underperform DQN early in training?**
A: With perfect hyperparameters, DQN can be slightly faster. Double DQN's advantage is stability and robustness.

**Q: Should I always use the largest buffer?**
A: No. Diminishing returns after 100k. Larger buffers waste memory without proportional benefits.

**Q: What if my environment differs from Lunar Lander?**
A: These findings are environment-specific. Apply the same analysis methodology to your environment.

**Q: How do I choose between hard and soft updates?**
A: Hard updates are simpler. Soft updates are smoother. For this environment, both work similarly at optimal settings.

**Q: Can I apply these findings to other control tasks?**
A: Yes, as a starting point. The general patterns (buffer size, exploration scheduling) apply widely, but specific values will differ by task.


