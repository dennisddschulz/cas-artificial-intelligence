import gymnasium as gym
import numpy as np
from collections import defaultdict
import mlflow
from pathlib import Path
from datetime import datetime


#%% mlflow setup
parent_folder = Path.cwd().parent
local_uri = Path("./0_mlflow_logs").absolute().as_uri()
mlflow.set_tracking_uri(local_uri)
mlflow.set_experiment("DRL TAXI IMPROVED")
from pathlib import Path
p = Path("./0_mlflow_logs").absolute()
print("folder:", p)
print("file URI:", f"file://{p}")

#%% Basic policies
def random_policy(s, nA):
    """Pure random policy."""
    return int(np.random.randint(nA))


def greedy_action_from_Q(Q, s, nA):
    """Return argmax_a Q(s,a) (ties -> first max)."""
    qs = [Q.get((s, a), 0.0) for a in range(nA)]
    return int(np.argmax(qs))


def make_epsilon_greedy_policy(Q, nA, eps, rng=None):
    """
    Epsilon-greedy policy from Q:
    - with probability eps choose random action
    - otherwise choose greedy(Q)
    """
    if rng is None:
        rng = np.random.default_rng()

    def policy(s, nA_ignored=None):
        if rng.random() < eps:
            return int(rng.integers(nA))
        return greedy_action_from_Q(Q, s, nA)

    return policy


#%% Episode rollout
def rollout_episode(env, policy_fn, max_steps=200, seed=None, record_frames=False):
    """
    Generate a single episode trajectory:
    traj = [(s, a, r), ...]
    """
    traj = []

    frames = [] if record_frames else None

    if seed is None:
        s, _ = env.reset()
    else:
        s, _ = env.reset(seed=seed)

    if record_frames:
        frames.append(env.render())

    for _ in range(max_steps):
        a = policy_fn(s, env.action_space.n)
        s2, r, terminated, truncated, _ = env.step(a)

        traj.append((s, a, r))
        s = s2

        # Capture the frame after the step
        if record_frames:
            frames.append(env.render())

        if terminated or truncated:
            break

    return (traj, frames) if record_frames else traj


#%% Monte Carlo incremental update (Q and V continuous)
def mc_update_QV_from_episode(
        traj,
        Q,
        V,
        N_sa,
        N_s,
        gamma=0.99,
        alpha_sa_min=0.01,
        alpha_s_min=0.01,
        first_visit=True
):
    """
    Incremental Monte Carlo update from ONE episode.

    Updates:
      - Q(s,a) using return G_t
      - V(s) using return G_t

    Using running mean:
      Q <- Q + (1/N) * (G - Q)
      V <- V + (1/N) * (G - V)
    """
    G = 0.0

    visited_sa = set()
    visited_s = set()

    # Backward return computation
    for t in reversed(range(len(traj))):
        s, a, r = traj[t]
        G = r + gamma * G

        # ---- Q(s,a) update ----
        if (not first_visit) or ((s, a) not in visited_sa):
            visited_sa.add((s, a))
            N_sa[(s, a)] += 1
            alpha_sa = max(1.0 / N_sa[(s,a)], alpha_sa_min) # learning rate
            Q[(s, a)] += alpha_sa * (G - Q[(s, a)])

        # ---- V(s) update ----
        if (not first_visit) or (s not in visited_s):
            visited_s.add(s)
            N_s[s] += 1
            alpha_s = max(1.0 / N_s[s], alpha_s_min) # learning rate!
            V[s] += alpha_s * (G - V[s])


#%% Taxi-v3 success rate evaluation (IMPORTANT!)
def evaluate_success_rate(env, policy_fn, episodes=1000, seed=0, max_steps=200):
    """
    Taxi-v3 success = correct drop-off.
    In Taxi-v3 the final reward for success is +20 on the terminal step.
    """
    successes = 0

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)

        for _ in range(max_steps):
            a = policy_fn(s, env.action_space.n)
            s, r, terminated, truncated, _ = env.step(a)

            if terminated or truncated:
                # Taxi-v3: successful dropoff => terminal +20 reward
                if terminated and r == 20:
                    successes += 1
                break

    return successes / episodes


#%% Epsilon schedules (if you want)
def epsilon_linear(ep, eps0=0.4, eps_min=0.05, decay_episodes=50_000):
    """Linear epsilon decay over decay_episodes."""
    frac = min(1.0, ep / decay_episodes)
    eps = eps0 + frac * (eps_min - eps0)
    return float(max(eps_min, eps))


def epsilon_exp(ep, eps0=0.4, eps_min=0.05, alpha=0.9995):
    """Exponential epsilon decay."""
    eps = eps0 * (alpha ** ep)
    return float(max(eps_min, eps))


#%% Full incremental MC control training loop
def train_mc_control_incremental(
        env,

        # general
        episodes=1_00_000,
        gamma=0.99, # how much the algo looks into the future!
        first_visit=False,

        # epsilon config
        eps_schedule="linear",     # "exp" or "linear"
        eps0=0.3, # randommness at the beginning (linear)
        eps_min=0.01, # 0.01, # randomnness in the end
        alpha=0.9995,           # for exp (curve for exp)
        decay_episodes=75_000,  # for linear

        # learning rate
        alpha_sa_min = 0.01,
        alpha_s_min = 0.01,

        # evaluation
        eval_every = 5000,
        eval_episodes = 10_000,

        # randommness
        seed=42
):

    # mlflow log params
    mlflow.log_params({
        # general
        "episodes": episodes,
        "gamma": gamma,
        "first_visit": first_visit,

        # epsilon config
        "eps_schedule": eps_schedule,
        "eps0": eps0,
        "eps_min": eps_min,
        "alpha": alpha,
        "decay_episodes": decay_episodes,

        # learning rate
        "alpha_sa_min": alpha_sa_min,
        "alpha_s_min": alpha_s_min,

        # evaluation
        "eval_every" : eval_every,
        "eval_episodes": eval_episodes,

        # randommness
        "seed": seed
    })

    """
    Incremental Monte Carlo Control:
    - continuously updates Q and V from episodes
    - policy improves gradually (epsilon-greedy from current Q)
    """
    rng = np.random.default_rng(seed)

    # global tables (persist across whole training)
    Q = defaultdict(float) # Q-values
    V = defaultdict(float) # V-values
    N_sa = defaultdict(int) # Number of visits for actions (s,a)
    N_s = defaultdict(int) # Number of visits state (s)

    # start policy
    eps = eps0
    policy = make_epsilon_greedy_policy(Q, env.action_space.n, eps, rng=rng)

    history = []
    s0, _ = env.reset(seed=seed)

    for ep in range(episodes):
        # rollout
        traj = rollout_episode(env, policy, max_steps=200, seed=None)

        # incremental update (continuous learning)
        mc_update_QV_from_episode(
            traj=traj,
            Q=Q,
            V=V,
            N_sa=N_sa,
            N_s=N_s,
            gamma=gamma,
            alpha_sa_min=alpha_sa_min,
            alpha_s_min=alpha_s_min,
            first_visit=first_visit
        )

        # update epsilon
        if eps_schedule == "exp":
            eps = epsilon_exp(ep, eps0=eps0, eps_min=eps_min, alpha=alpha)
        elif eps_schedule == "linear":
            eps = epsilon_linear(ep, eps0=eps0, eps_min=eps_min, decay_episodes=decay_episodes)
        else:
            raise ValueError("eps_schedule must be 'exp' or 'linear'")

        # refresh policy (epsilon-greedy from updated Q)
        policy = make_epsilon_greedy_policy(Q, env.action_space.n, eps, rng=rng)

        # evaluate
        if (ep % eval_every == 0) or (ep == episodes - 1):
            sr = evaluate_success_rate(env, policy, episodes=eval_episodes, seed=seed + 12345)
            v0 = float(V.get(s0, 0.0))

            history.append({
                "episode": ep,
                "eps": eps,
                "V(start)": v0,
                "success_rate": sr
            })

            print(f"ep={ep:6d} | eps={eps:.3f} | V(start)={v0:.3f} | success_rate={sr:.3f}")

            # get the maximum q values
            q_values = list(Q.values())
            q_values_mean = np.mean(q_values)
            q_values_max = np.max(q_values)
            q_values_min = np.min(q_values)
            q_values_std = np.std(q_values)

            # log mlflow metrics
            mlflow.log_metric("epsilon", eps, step=ep)
            mlflow.log_metric("V_start", v0, step=ep)
            mlflow.log_metric("success_rate", sr, step=ep)
            mlflow.log_metric("q-value_mean", q_values_mean, step=ep)
            mlflow.log_metric("q_values_max", q_values_max, step=ep)
            mlflow.log_metric("q-values_min", q_values_min, step=ep)
            mlflow.log_metric("q-values_std", q_values_std, step=ep)

    return policy, Q, V, history


#%% Play / test learned policy
def play_policy(env, policy_fn, episodes=10, seed=0, max_steps=200):
    """
    Run episodes and print rewards + success count (debug helper).
    """
    successes = 0
    total_rewards = []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        ep_reward = 0

        for _ in range(max_steps):
            a = policy_fn(s, env.action_space.n)
            s, r, terminated, truncated, _ = env.step(a)
            ep_reward += r

            if terminated or truncated:
                if terminated and r == 20:
                    successes += 1
                break

        total_rewards.append(ep_reward)

    print(f"Played {episodes} episodes | successes={successes}/{episodes} | avg_reward={np.mean(total_rewards):.2f}")
    return total_rewards


#%% calls
env = gym.make("Taxi-v3", render_mode="rgb_array")

# --- mlflow run ---
print("policy improvement start!")
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
mlflow.end_run()  # ends the currently active run
mlflow.start_run(run_name=run_name)

# train the model!
trained_policy, Q, V, hist = train_mc_control_incremental(env)

# play the model (test!)
play_policy(env, trained_policy, episodes=50, seed=999)


#%% visualize the trajectory
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# execute a taxi ride for the plot!
traj, frames = rollout_episode(env, trained_policy, max_steps=200, seed=None, record_frames=True)
# traj = list of (state, action, reward)
# env = your Gymnasium environment

# Create figure for animation
fig, ax = plt.subplots()
ax.axis('off')
im = ax.imshow(frames[0])

def update(frame):
    im.set_data(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=500, blit=True)
plt.show()


#%% export trajectory to mlflow
import json

# traj = [(s, a, r), ...]
# Convert tuples to lists for JSON
traj_json = [list(step) for step in traj]

# Save to file
traj_file = "traj.json"
with open(traj_file, "w") as f:
    json.dump(traj_json, f, indent=2)  # indent makes it readable

# Log as artifact to MLflow
mlflow.log_artifact(traj_file, artifact_path="trajectories")



#%% decode and export trajectory as csv
import pandas as pd

def decode_taxi_state(state):
    """
    Returns taxi_row, taxi_col, passenger_loc, destination
    passenger_loc: 0-3=at R/G/Y/B, 4=In Taxi
    dest: 0-3=R/G/Y/B
    fomula = state = ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    """
    taxi_row = state // 100 # 5 * 5 * 24 * 4 = 100
    state %= 100
    taxi_col = state // 20 # 5 * 4 = 20
    state %= 20
    passenger_loc = state // 4 # 4
    dest = state % 4
    return taxi_row, taxi_col, passenger_loc, dest


locations_full = {0: "Red", 1: "Green", 2: "Yellow", 3: "Blue", 4: "In Taxi"}
actions_map = ["South", "North", "East", "West", "Pickup", "Dropoff"]

def traj_to_dataframe(traj):
    rows = []
    for i, (state, action, reward) in enumerate(traj):
        row, col, passenger_loc, dest = decode_taxi_state(state)
        rows.append({
            "Step": i+1,
            "Taxi Row": row,
            "Taxi Col": col,
            "Passenger Location": locations_full.get(passenger_loc),
            "Destination": locations_full.get(dest),
            "Action": actions_map[action],
            "Reward": reward
        })
    return pd.DataFrame(rows)

df_traj = traj_to_dataframe(traj)
print(df_traj)
df_traj.to_csv("traj.csv", index=False)               # save CSV
mlflow.log_artifact("traj.csv", artifact_path="trajectories")  # log CSV


#%% save Q values to mlflow
import csv
with open("q_table.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Header
    writer.writerow(["state", "action", "value"])

    # Write Q entries
    for (s, a), value in Q.items():
        writer.writerow([s, a, value])
mlflow.log_artifact("q_table.csv", artifact_path="values")  # log CSV

#%% save V values to mlflow
import csv
with open("v_table.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Header
    writer.writerow(["state", "value"])

    # Write V entries
    for s, value in V.items():
        writer.writerow([s, value])
mlflow.log_artifact("v_table.csv", artifact_path="values")  # log CSV


# mlflow done!
mlflow.end_run()
