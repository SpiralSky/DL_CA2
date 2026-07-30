# %% [markdown]
# # Part B: Reinforcement Learning — DQN on Pendulum-v0
#
# **ST1504 Deep Learning — CA2 Part B (45 marks)**
#
# **Task.** Apply a Deep Q-Network (DQN) to balance the OpenAI Gym `Pendulum-v0` environment. Since DQN requires a *discrete* action space and Pendulum's action space is continuous, we discretise the torque. We investigate four gravity configurations:
#
# | # | Setting | `g` value |
# |---|---|---|
# | 1 | Default gravity | `10.0` |
# | 2 | Free-fall | `0.0` |
# | 3 | Anti-gravity | `-10.0` |
# | 4 | Supergravity | `15.0` |
#
# **Roadmap of this notebook**
# 1. Environment setup & exploration
# 2. Action discretisation
# 3. Gravity configurations — a physics-first preliminary investigation
# 4. DQN architecture (Q-network, target network, replay buffer)
# 5. Training loop & experimental design (why multiple seeds, episode budget)
# 6. Results — learning curves across all 4 settings
# 7. Systematic evaluation — defining "best" quantitatively
# 8. Model improvement — an ablation study proving the target network / replay buffer matter
# 9. Saving the best model
# 10. Visualising the trained agent — an embedded video, with a mid-episode disturbance test
# 11. Summary & conclusions
#
# > **Compute constraint.** All training below runs on a single CPU core, no GPU — episode counts are calibrated to finish in a few minutes while still showing genuine learning. Section 5 explains the exact numbers and measured training time. For a longer run (more seeds, more episodes), only the `N_EPISODES` / `N_SEEDS` constants need to change.

# %%
# gym 0.17.3 is not preinstalled on Colab (which ships gymnasium, or nothing) -- install the
# exact pinned version the assignment brief specifies, matching the CartPole lab.
# !pip install "gym==0.17.3" -q

# %% [markdown]
# # Background Research
#
# **Problem:**
#
# The Pendulum environment is a continuous-control reinforcement learning problem where an agent learns to apply torque to swing the pendulum upright and maintain balance. The objective is to maximise cumulative reward by minimising the pendulum's angle from the upright position, angular velocity, and unnecessary control effort.
#
# Deep Q-Network (DQN) is a value-based reinforcement learning algorithm capable of approximating Q-values using a neural network. However, standard DQN only supports discrete action spaces, whereas the Pendulum environment requires continuous torque outputs.
#
# Therefore, the first challenge is adapting DQN to solve a continuous-control problem.
#
# Approaches taken:
#
# 1. Action Discretisation
# 2. Improved DQN Architecture
# 3. Multiple Gravity Configurations
# 4. Repeated Experiments

# %%
import random
import time
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

import gym
print("gym version:", gym.__version__, " (pinned to 0.17.3 per assignment brief, matching the CartPole lab)")
print("TensorFlow version:", tf.__version__)

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
tf.random.set_seed(GLOBAL_SEED)

# %% [markdown]
# ## 1. Environment Setup & Exploration
#
# `Pendulum-v0` is a classic continuous-control benchmark: a frictionless pendulum starts in a random position and the agent must apply torque to swing it up and hold it upright.
#
# - **State** (continuous, 3-dim): `[cos(theta), sin(theta), theta_dot]` — the pendulum angle is given as its cosine/sine (to avoid the discontinuity at the ±π wrap-around) plus angular velocity.
# - **Action** (continuous, 1-dim): torque in `[-2.0, 2.0]` N·m.
# - **Reward**: `-(theta_normalised^2 + 0.1*theta_dot^2 + 0.001*action^2)` at every step — it is always ≤ 0, and is maximised (closest to 0) when the pendulum is upright (`theta=0`), nearly still, and using little torque. There is no terminal state; every episode simply runs for a fixed 200 steps.

# %%
env = gym.make("Pendulum-v0")
print("Action space:", env.action_space, " -> continuous torque, 1-dim")
print("Observation space:", env.observation_space, " -> [cos(theta), sin(theta), theta_dot]")
print("Max episode steps:", env._max_episode_steps)

env.seed(GLOBAL_SEED)
obs = env.reset()
print("\nExample reset observation:", obs)
obs, reward, done, info = env.step(env.action_space.sample())
print("Example step  -> obs:", obs, " reward:", round(reward, 3), " done:", done)
env.close()

# %% [markdown]
# Pendulum-v0 is officially documented as an **unsolved environment** — it has no standard "reward threshold" for being considered solved (unlike e.g. CartPole). This matters for Section 7: rather than using a borrowed number, the `-400` threshold used there is self-defined and justified there directly.

# %%
env2 = gym.make("Pendulum-v0")
print("env.spec.id:", env2.spec.id)
print("env.spec.max_episode_steps:", env2.spec.max_episode_steps)
print("env.spec.reward_threshold:", env2.spec.reward_threshold)
print("env.spec.nondeterministic:", env2.spec.nondeterministic)
env2.close()

# %% [markdown]
# ## 2. Action Discretisation
#
# DQN outputs one Q-value per *discrete* action, so it cannot directly handle the continuous torque space. As covered in the Topic 6 lecture's discussion of discretising continuous states for Q-tables, there is a direct trade-off: more bins → finer control but a larger output layer and a harder function to learn; fewer bins → coarser control but faster, more stable learning.
#
# We discretise torque into **9 evenly-spaced bins** across `[-2.0, 2.0]` (including 0, so the agent can choose "no torque"). Nine bins is a middle ground that keeps the network small while still giving the agent enough resolution to make smooth, graded corrections rather than only full-strength ±2.0 bang-bang control.

# %%
N_BINS = 9
ACTION_BINS = np.linspace(-2.0, 2.0, N_BINS)
N_ACTIONS = len(ACTION_BINS)
print("Discrete action set (torque, N\u00b7m):", np.round(ACTION_BINS, 3))


# %% [markdown]
# ## 3. Gravity Configurations — A Physics-First Preliminary Investigation
#
# Gravity is set via `env.unwrapped.g` immediately after `gym.make(...)` (confirmed to actually change the simulated dynamics, not just a cosmetic flag). Before training anything, it's worth asking: what does each `g` value actually *do* to the physics? Looking at the environment's own update equation,
#
# ```
# new_theta_dot = theta_dot + (3*g / (2*l) * sin(theta) + 3/(m*l^2) * torque) * dt
# ```
#
# the gravity term `3g/(2l)*sin(theta)` acts as a torque that is **zero at theta=0 (upright)** and grows with the tilt angle. Its *sign* determines whether it pushes the pendulum further from upright (destabilising) or back toward upright (self-righting):
#
# - `g > 0` → destabilising at upright (the classic "inverted pendulum" problem — gravity actively fights the agent).
# - `g < 0` → **stabilising** at upright (the sign flip means gravity now pulls *toward* the goal state instead of away from it).
# - `g = 0` → no gravitational torque at all; only momentum and the agent's own torque matter.
#
# We can check this directly with zero torque applied, starting from a small tilt, and watching whether the angle drifts away from upright or back toward it — this gives us a testable hypothesis about difficulty *before* spending any training time.

# %%
def zero_torque_trace(g_val, start_theta=0.2, n_steps=40):
    env = gym.make("Pendulum-v0")
    env.reset()
    env.unwrapped.g = g_val
    env.unwrapped.state = np.array([start_theta, 0.0])
    thetas = [start_theta]
    for _ in range(n_steps):
        _, _, _, _ = env.step(np.array([0.0]))  # apply NO torque, physics only
        thetas.append(env.unwrapped.state[0])
    env.close()
    return thetas

GRAVITY_SETTINGS = {
    "Default (g=10)": 10.0,
    "Free-fall (g=0)": 0.0,
    "Anti-gravity (g=-10)": -10.0,
    "Supergravity (g=15)": 15.0,
}

plt.figure(figsize=(9, 5))
for label, g_val in GRAVITY_SETTINGS.items():
    trace = zero_torque_trace(g_val)
    plt.plot(trace, marker="o", markersize=3, label=label)
plt.axhline(0, color="black", linewidth=0.8, linestyle="--", label="upright (goal)")
plt.xlabel("simulation step (no torque applied)")
plt.ylabel("theta (rad), 0 = upright")
plt.title("Does each gravity setting pull the pendulum toward or away from upright?")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("gravity_preliminary_investigation.png", dpi=110)
plt.show()


# %% [markdown]
# **Reading the plot / hypothesis going in:** with zero torque, the default and supergravity settings visibly drift *away* from `theta=0` (destabilising, and faster for `g=15`), free-fall stays essentially flat (neutral), and anti-gravity visibly drifts *back toward* `theta=0` (self-stabilising). Our working hypothesis is therefore an easy→hard ordering of roughly **anti-gravity ≈ free-fall < default < supergravity**. We test this hypothesis against the actual DQN learning curves in Section 6-7 — the training results, not this preliminary check, are what decide the final ranking.

# %% [markdown]
# ## 4. DQN Architecture
#
# Following the two key ingredients highlighted in the Topic 6 lecture for extending Q-learning from discrete Q-tables to continuous state spaces:
#
# 1. **Experience replay buffer** — recent transitions `(state, action, reward, next_state, done)` are stored in a fixed-size buffer (we use 10,000, matching the lecture's example capacity) and sampled in random mini-batches (32 at a time, also matching the lecture). This breaks the strong correlation between consecutive Pendulum steps, which otherwise destabilises training.
# 2. **Target network** — a second copy of the Q-network, frozen most of the time, is used to compute the bootstrapped target `r + γ·max_a' Q_target(s', a')`. It is synced to the main network's weights periodically (every 5 episodes here). Without this, the network is chasing a constantly-moving target defined by itself, which is a well-known source of divergence.
#
# **Q-network:** a simple MLP, `state(3) → Dense(64, relu) → Dense(64, relu) → Dense(9, linear)`, one output per discretised action. Kept deliberately small — Pendulum's state space is low-dimensional, and a small network trains faster per step, which matters a lot on CPU-only compute.
#
# **Exploration:** epsilon-greedy, starting at `epsilon=1.0` (fully random) and decaying by a factor of 0.97 each episode down to a floor of `0.05`, so the agent explores heavily early on and mostly exploits its learned policy later while still taking occasional random actions.
#
# **Implementation note on speed:** an early prototype that called Keras `.predict()`/`.fit()` at every single environment step took **~40 seconds per episode** — far too slow to compare 4 settings. Switching to direct `@tf.function`-wrapped model calls for inference and `train_on_batch()` for updates, and training only every 4 steps instead of every step, brought this down to **~0.2-0.24 seconds per episode** (~150-170x faster) with no loss in learning quality. This is the version implemented below.

# %%
class DQNAgent:
    def __init__(self, state_dim, n_actions, gamma=0.98, lr=1e-3,
                 buffer_size=10_000, batch_size=32, train_every=4):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.train_every = train_every
        self.buffer = deque(maxlen=buffer_size)
        self.step_count = 0

        self.q_net = self._build_network(lr)
        self.target_net = self._build_network(lr)
        self.target_net.set_weights(self.q_net.get_weights())

    def _build_network(self, lr):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(self.n_actions, activation="linear"),
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
        return model

    @staticmethod
    @tf.function
    def _forward(model, states):
        return model(states, training=False)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        q_vals = self._forward(self.q_net, state[None, :].astype(np.float32)).numpy()[0]
        return int(np.argmax(q_vals))

    def remember(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))
        self.step_count += 1

    def maybe_train(self):
        if len(self.buffer) < self.batch_size or self.step_count % self.train_every != 0:
            return
        batch = random.sample(self.buffer, self.batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        max_next_q = np.max(self._forward(self.target_net, next_states).numpy(), axis=1)
        targets = self._forward(self.q_net, states).numpy()
        targets[np.arange(self.batch_size), actions] = rewards + self.gamma * max_next_q * (1 - dones)
        self.q_net.train_on_batch(states, targets)

    def update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def save(self, path):
        self.q_net.save(path)


# %% [markdown]
# ## 5. Training Loop & Experimental Design
#
# **Why multiple seeds?** DQN training is noisy — random network initialisation, random replay sampling, and epsilon-greedy exploration all mean a *single* training run can get lucky or unlucky. One run finishing strong doesn't tell us whether a gravity setting is genuinely easier, or whether we just had a good seed. We therefore train **2 independent seeds per gravity setting** (8 runs total) and report both the mean and the spread across seeds. Two is a compute-budget compromise for this notebook — for the final submission we'd recommend 3-5 seeds per setting on a GPU runtime for tighter confidence.
#
# **Episode budget:** `N_EPISODES = 180` per run. This was calibrated empirically on this sandbox's single CPU core: at ~0.2-0.24s/episode, 180 episodes takes well under a minute per run, and an early prototype showed the reward curve clearly bending toward convergence by episode 150, so 180 episodes is enough to show genuine learning without an excessive wait. This constant is the one thing to raise (e.g. to 400-600) when re-running with more compute — the code below records exactly how long training actually took, so that trade-off is measured, not guessed.
#
# **On rendering during training vs. after.** The CartPole DQN lab renders periodically *inside* the training loop itself:
# ```python
# if not _ % ShowEvery and len(DQN.ReplayMemory) >= DQN.MinReplayMemory:
#     env.render()
# ```
# with `ShowEvery = 10` — a popup every 10th episode, showing the *current, still-learning* policy. `train_dqn` below supports the same pattern via an optional `render_every` argument, for exactly this reason. It's **off by default for the 8-run sweep below** (4 gravity settings × 2 seeds): the lab renders one environment, once; this notebook trains 8 configurations back-to-back specifically to compare them, and periodic popups on all 8 would both be impractical (8x the interruptions) and silently do nothing anyway on Colab/headless (same reason explained in Section 10). Section 10 instead renders **after** training, from the saved best model — a deliberate choice, not an oversight, but the in-loop option is genuinely there in the code below if you want to watch a specific run learn live on a local machine.

# %%
N_EPISODES = 180
N_SEEDS = 2
SEEDS = [0, 1]
MAX_STEPS = 200  # matches Pendulum-v0's built-in TimeLimit

def train_dqn(gravity, seed, n_episodes=N_EPISODES, verbose_every=60, render_every=None):
    """render_every: if set (e.g. 10, matching the CartPole lab's ShowEvery), attempts a live
    env.render() popup every N episodes -- same graceful fallback as Section 10 if no display."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    env = gym.make("Pendulum-v0")
    env.seed(seed)
    env.unwrapped.g = gravity

    agent = DQNAgent(state_dim=3, n_actions=N_ACTIONS)
    epsilon, eps_min, eps_decay = 1.0, 0.05, 0.97

    reward_history = []
    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        show_this_episode = render_every and (ep % render_every == 0) and len(agent.buffer) >= agent.batch_size
        for _ in range(MAX_STEPS):
            a_idx = agent.act(state, epsilon)
            action = np.array([ACTION_BINS[a_idx]])
            if show_this_episode:
                try:
                    env.render(mode="human")
                except Exception:
                    show_this_episode = False  # no display available; stop trying for the rest of this run
            next_state, reward, done, info = env.step(action)
            agent.remember(state, a_idx, reward, next_state, done)
            agent.maybe_train()
            state = next_state
            ep_reward += reward
        if ep % 5 == 0:
            agent.update_target()
        epsilon = max(eps_min, epsilon * eps_decay)
        reward_history.append(ep_reward)
        if (ep + 1) % verbose_every == 0:
            print(f"    ep {ep+1:>3}/{n_episodes}  last20_avg={np.mean(reward_history[-20:]):>8.1f}  eps={epsilon:.2f}")
    env.close()
    return agent, reward_history

results = {}      # {gravity_label: {seed: reward_history}}
trained_agents = {}  # {gravity_label: {seed: agent}}  (kept for weight-saving later)

t_start = time.time()
for label, g_val in GRAVITY_SETTINGS.items():
    print(f"\n=== Training: {label} ===")
    results[label] = {}
    trained_agents[label] = {}
    for seed in SEEDS:
        print(f"  seed {seed}:")
        agent, hist = train_dqn(g_val, seed)  # render_every left off for this comparison sweep -- see note above
        results[label][seed] = hist
        trained_agents[label][seed] = agent
print(f"\nTotal training time for {len(GRAVITY_SETTINGS)} settings x {N_SEEDS} seeds x {N_EPISODES} episodes: {time.time()-t_start:.1f}s")

# %% [markdown]
# ## 6. Results — Learning Curves

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
colors = plt.cm.tab10.colors

for ax, (label, seed_histories) in zip(axes.flat, results.items()):
    for i, (seed, hist) in enumerate(seed_histories.items()):
        # 10-episode moving average for readability
        smoothed = pd.Series(hist).rolling(10, min_periods=1).mean()
        ax.plot(smoothed, color=colors[i], alpha=0.9, label=f"seed {seed}")
        ax.plot(hist, color=colors[i], alpha=0.15)
    ax.set_title(label)
    ax.set_xlabel("episode")
    ax.set_ylabel("episode reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle("DQN learning curves by gravity setting (faint = raw reward, solid = 10-ep moving avg)", y=1.01)
plt.tight_layout()
plt.savefig("learning_curves_by_gravity.png", dpi=110, bbox_inches="tight")
plt.show()

# overlay comparison: mean across seeds per setting
plt.figure(figsize=(9, 5.5))
for i, (label, seed_histories) in enumerate(results.items()):
    mean_curve = pd.DataFrame(seed_histories).mean(axis=1)
    smoothed = mean_curve.rolling(10, min_periods=1).mean()
    plt.plot(smoothed, label=label, color=colors[i], linewidth=2)
plt.xlabel("episode")
plt.ylabel("episode reward (mean across seeds, 10-ep moving avg)")
plt.title("Learning curves overlaid — all 4 gravity settings")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curves_overlay.png", dpi=110)
plt.show()


# %% [markdown]
# ## 7. Systematic Evaluation — Defining "Best" Quantitatively
#
# "Best" is ambiguous unless we pin down what we're optimising for, so we score each gravity setting on three separate criteria:
#
# 1. **Final performance** — mean episode reward over the *last 20 episodes*, averaged across both seeds. Higher (closer to 0) is better. This answers "how good is the converged policy?"
# 2. **Stability** — standard deviation of that same final-20-episode reward, averaged across seeds. Lower is better. This answers "how consistent/reliable is the agent once trained, episode to episode?"
# 3. **Learning speed** — the first episode at which the 10-episode moving average reward crosses a fixed threshold (`-400`, a level clearly better than random-policy performance but not requiring full convergence). Lower is better. This answers "how quickly does a usable policy emerge?" Pendulum-v0 has no official "solved" threshold (Section 1), so `-400` is self-set and explicit rather than an unverified borrowed number.
#
# No single number captures "best" — a setting could converge fast but to a worse final policy, or converge slowly but very stably. We report all three rather than collapsing to one score.

# %%
def episodes_to_threshold(hist, threshold=-400, window=10):
    smoothed = pd.Series(hist).rolling(window, min_periods=1).mean()
    reached = smoothed[smoothed >= threshold]
    return int(reached.index[0]) + 1 if len(reached) > 0 else None

rows = []
for label, seed_histories in results.items():
    finals = [np.mean(hist[-20:]) for hist in seed_histories.values()]
    stds = [np.std(hist[-20:]) for hist in seed_histories.values()]
    speeds = [episodes_to_threshold(hist) for hist in seed_histories.values()]
    speeds_valid = [s for s in speeds if s is not None]
    rows.append({
        "Gravity setting": label,
        "Final performance (mean reward, last 20 ep)": round(np.mean(finals), 1),
        "Stability (std of reward, last 20 ep)": round(np.mean(stds), 1),
        f"Episodes to reach reward >= -400": (round(np.mean(speeds_valid), 1) if speeds_valid else "not reached"),
    })

summary_df = pd.DataFrame(rows).set_index("Gravity setting")
summary_df

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

labels = list(results.keys())
finals = [summary_df.loc[l, "Final performance (mean reward, last 20 ep)"] for l in labels]
stds = [summary_df.loc[l, "Stability (std of reward, last 20 ep)"] for l in labels]

axes[0].bar(labels, finals, color=colors[:4])
axes[0].set_ylabel("mean reward, last 20 episodes")
axes[0].set_title("Final performance by gravity setting")
axes[0].tick_params(axis="x", rotation=20)
axes[0].grid(alpha=0.3, axis="y")

axes[1].bar(labels, stds, color=colors[:4])
axes[1].set_ylabel("std of reward, last 20 episodes")
axes[1].set_title("Stability by gravity setting (lower = more consistent)")
axes[1].tick_params(axis="x", rotation=20)
axes[1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("final_performance_comparison.png", dpi=110)
plt.show()


# %% [markdown]
# **Interpreting the results.**
#
# The three criteria tell a strikingly consistent story that matches the Section 3 hypothesis almost exactly: **free-fall and anti-gravity form a clearly easier cluster than default and supergravity, on every single metric.**
#
# | Setting | Final performance | Stability (std) | Episodes to threshold |
# |---|---|---|---|
# | Free-fall (g=0) | **-45.3** (best) | **36.1** (best) | 23.5 |
# | Anti-gravity (g=-10) | -56.7 | 47.5 | **16.0** (best) |
# | Default (g=10) | -283.9 | 262.7 | 99.0 |
# | Supergravity (g=15) | -321.1 (worst) | 292.9 (worst) | 126.0 (worst) |
#
# - **The easy pair vs. the hard pair.** Free-fall and anti-gravity finish roughly 5-7x better than default and supergravity, with 6-8x lower variance, and reach a usable policy 4-8x faster. This confirms that what matters most for how hard this task is for DQN isn't gravity's *magnitude* — it's whether gravity actively fights the agent (`g>0`, destabilising at upright) or not (`g≤0`).
# - **A nuance within the easy pair.** Anti-gravity reaches the -400 threshold fastest (16 episodes vs. 23.5) — its self-righting physics means even a mediocre early policy tends to drift toward upright "for free," so *some* reward quickly follows. But free-fall ultimately edges it out on final performance and stability. A plausible explanation: anti-gravity's restoring force behaves like a lightly-damped spring around upright — it pulls the pendulum back, but can also make it overshoot and oscillate unless the agent learns fine torque control to actively damp that swing. Free-fall has no such ambient force at all: once the agent learns to zero out its own momentum near upright, nothing is left to disturb it, which allows the tightest and most consistent convergence of the four settings.
# - **A nuance within the hard pair.** Supergravity is uniformly worse than default gravity on all three criteria, consistent with it being "the same destabilising direction, just stronger." Since the torque cap (±2.0 N·m) doesn't scale up to compensate, larger tilt angles under `g=15` likely become genuinely unrecoverable within the available torque budget — not just harder to learn, but occasionally physically un-counterable — which also explains its much larger reward variance (292.9, the highest of the four).
# - **So, which is "best"?** By this notebook's definition (final performance, then stability, then speed as tie-breakers), **free-fall (g=0)** is the best setup — it wins 2 of 3 criteria outright and is a close second on the third. If "fastest to a usable policy" were the priority instead, anti-gravity would be the pick. Either answer is defensible; what matters is that the criteria — and the trade-off between them — are stated explicitly rather than picking whichever setting "looks good" after the fact.
# - **Bigger picture.** This is a useful reminder that RL difficulty doesn't track human intuition about "which gravity setting sounds scariest." An agent doesn't care whether physics is helping it (anti-gravity), staying out of the way (free-fall), or fighting it (default, supergravity) in any qualitative sense — what it actually has to overcome is the *size and directional consistency* of the disturbance forcing it away from the goal, and that is exactly what separates the two clusters above.

# %% [markdown]
# ## 8. Model Improvement — Does the Target Network / Replay Buffer Actually Help?
#
# Section 4 explained *why* the Topic 6 lecture says these two ingredients matter, quoting the lecture directly. But quoting a claim isn't the same as demonstrating it. The brief explicitly asks us to "improve the RL agent's performance systematically" — so rather than just asserting the target network and replay buffer help, we ran a controlled **ablation study**: train the exact same DQN, on the exact same gravity setting (default, g=10), with the exact same hyperparameters, and remove one ingredient at a time.
#
# - **Full DQN**: target network + 10,000-transition replay buffer (as used throughout this notebook).
# - **No target network**: bootstraps `max Q(s')` off the *live, currently-training* network instead of a frozen copy — this is exactly the failure mode the lecture warns about (“trying to train a complex neural network and asking it to make predictions usually lead to unstable training... can just diverge”).
# - **No replay buffer**: trains on only the single most recent transition each step (batch size 1, no random sampling) instead of a shuffled batch from history — directly re-introduces the correlated-consecutive-steps problem replay buffers are meant to fix.
#
# Everything else (network size, learning rate, epsilon schedule, episode budget) is held identical across all three, so any performance difference is attributable to the ingredient removed, not a confound.

# %%
def build_network(state_dim, n_actions, lr=1e-3):
    model = tf.keras.Sequential([
        layers.Input(shape=(state_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_actions, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def train_ablation(condition, seed, n_episodes=150):
    """condition: 'full', 'no_target', or 'no_replay'."""
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    use_target = condition != "no_target"
    use_replay = condition != "no_replay"

    q_net = build_network(3, N_ACTIONS)
    target_net = build_network(3, N_ACTIONS)
    target_net.set_weights(q_net.get_weights())

    buffer = deque(maxlen=10_000 if use_replay else 1)
    batch_size = 32 if use_replay else 1

    env = gym.make("Pendulum-v0")
    env.seed(seed)
    env.unwrapped.g = 10.0  # default gravity: representative single setting for this ablation

    epsilon, eps_min, eps_decay = 1.0, 0.05, 0.97
    step_count = 0
    reward_history = []
    for ep in range(n_episodes):
        state = env.reset()
        ep_reward = 0.0
        for _ in range(200):
            step_count += 1
            if random.random() < epsilon:
                a_idx = random.randrange(N_ACTIONS)
            else:
                q_vals = q_net(state[None, :].astype(np.float32), training=False).numpy()[0]
                a_idx = int(np.argmax(q_vals))
            action = np.array([ACTION_BINS[a_idx]])
            next_state, reward, done, info = env.step(action)
            buffer.append((state, a_idx, reward, next_state, done))
            state = next_state
            ep_reward += reward

            if len(buffer) >= batch_size and step_count % 4 == 0:
                batch = random.sample(buffer, batch_size) if use_replay else [buffer[-1]]
                states = np.array([b[0] for b in batch], dtype=np.float32)
                actions = np.array([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch], dtype=np.float32)
                next_states = np.array([b[3] for b in batch], dtype=np.float32)
                dones = np.array([b[4] for b in batch], dtype=np.float32)
                bootstrap_net = target_net if use_target else q_net
                max_next_q = np.max(bootstrap_net(next_states, training=False).numpy(), axis=1)
                targets = q_net(states, training=False).numpy()
                targets[np.arange(len(batch)), actions] = rewards + 0.98 * max_next_q * (1 - dones)
                q_net.train_on_batch(states, targets)

        if use_target and ep % 5 == 0:
            target_net.set_weights(q_net.get_weights())
        epsilon = max(eps_min, epsilon * eps_decay)
        reward_history.append(ep_reward)
    env.close()
    return np.array(reward_history)


ablation_results = {}
for condition in ["full", "no_target", "no_replay"]:
    ablation_results[condition] = {}
    for seed in [0, 1]:
        ablation_results[condition][seed] = train_ablation(condition, seed)
        print(f"{condition:10s} seed {seed}: last-20-episode avg reward = "
              f"{ablation_results[condition][seed][-20:].mean():.1f}")

# %%
ablation_labels = {
    "full": "Full DQN\n(replay buffer + target net)",
    "no_target": "No target network",
    "no_replay": "No replay buffer",
}
ablation_colors = {"full": "#3B82C4", "no_target": "#D97706", "no_replay": "#B3261E"}

plt.figure(figsize=(9.5, 5.5))
for condition, label in ablation_labels.items():
    mean_curve = pd.DataFrame(ablation_results[condition]).mean(axis=1)
    smoothed = mean_curve.rolling(10, min_periods=1).mean()
    plt.plot(smoothed, label=label.split("\n")[0], color=ablation_colors[condition], linewidth=2)
plt.xlabel("episode"); plt.ylabel("episode reward (mean across 2 seeds, 10-ep moving avg)")
plt.title("Ablation study: does the target network / replay buffer actually help?")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ablation_learning_curves.png", dpi=110)
plt.show()

means = [np.mean([ablation_results[c][s][-20:].mean() for s in [0, 1]]) for c in ablation_labels]
stds = [np.std([ablation_results[c][s][-20:].mean() for s in [0, 1]]) for c in ablation_labels]
plt.figure(figsize=(8.5, 5))
bars = plt.bar([l.split(chr(10))[0] for l in ablation_labels.values()], means,
               color=list(ablation_colors.values()), yerr=stds, capsize=6)
plt.ylabel("mean reward, last 20 episodes (higher = better)")
plt.title("Final performance with each DQN ingredient removed")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("ablation_bar_summary.png", dpi=110)
plt.show()

# %% [markdown]
# **Result: both ingredients matter, a lot — this isn't a marginal effect.**
#
# | Condition | Mean final reward (last 20 ep) |
# |---|---|
# | Full DQN | **-288** |
# | No target network | -1225 |
# | No replay buffer | -1485 |
#
# Removing *either* ingredient leaves the agent barely better than its own random starting point (~-1350 to -1380 average reward in the first 10 episodes) — both ablated variants essentially fail to learn a useful policy within the same 150-episode budget the full version uses to reach ~-288. No replay buffer is the more damaging removal here (mean -1485 vs -1225), consistent with batch-size-1 training on highly correlated consecutive pendulum states being a particularly unstable regime — exactly the failure mode the lecture's "uncorrelated datapoints → stable training" claim predicts.
#
# This is what "systematically improving the agent" means in practice for us: not just tuning numbers, but empirically verifying which architectural choices are load-bearing before trusting them in the full 4-gravity-setting comparison in Sections 6-7.

# %% [markdown]
# ## 9. Saving the Best Model

# %%
# "Best" here = highest final-performance score from Section 7's table
best_label = summary_df["Final performance (mean reward, last 20 ep)"].idxmax()
best_seed = max(results[best_label], key=lambda s: np.mean(results[best_label][s][-20:]))
best_agent = trained_agents[best_label][best_seed]

print(f"Best configuration: {best_label}, seed {best_seed}, "
      f"final performance = {np.mean(results[best_label][best_seed][-20:]):.1f}")

best_agent.save("best_dqn_pendulum.h5")
print("Saved best_dqn_pendulum.h5")

# also save one weights file per gravity setting (best seed of each) for reproducibility/comparison
import os
os.makedirs("all_gravity_weights", exist_ok=True)
for label in results:
    seed_for_label = max(results[label], key=lambda s: np.mean(results[label][s][-20:]))
    fname = "all_gravity_weights/" + label.split(" ")[0].lower() + f"_seed{seed_for_label}.h5"
    trained_agents[label][seed_for_label].save(fname)
    print("Saved", fname)

# %% [markdown]
# ## 10. Visualising the Trained Agent
#
# Reward numbers and learning curves prove the agent works, but they don't let you *see* it. This renders the trained policy as an actual animation: the pendulum swings up under the agent's control, and — to make the demo more than "it swings up once and sits there" — we apply two hard external shoves mid-episode (steps 90 and 170, bob turns orange) so you can watch the controller actively recover, not just hold a fixed starting position. A curved green arrow around the pivot shows the torque direction and strength at every step — which way, and how hard, the agent is turning it to balance.
#
# **Two display modes, tried automatically in order:**
# 1. **Live popup window** — `env.render(mode="human")`. Only works with a real display attached (a local machine with a screen) — it needs pyglet/OpenGL and a window manager, neither of which exist on Colab or a headless server.
# 2. **Embedded video (automatic fallback)** — if the popup can't open (caught explicitly below, not a silent failure), the same animation is built in matplotlib instead and embedded as a real HTML5 video. This always works, in any environment, and is what actually renders when this notebook is opened, submitted, or exported to HTML.
#
# **On speed:** the full 260-step rollout is played at 2x (every 2nd simulation step is shown, at 30fps instead of 20), cutting playback from 13s to about 4s without cutting either disturbance out of the story.

# %%
import time
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from IPython.display import HTML

def rollout_with_disturbance(model_path, gravity, seed=3, max_steps=260, disturbance_steps=(90, 170)):
    """Runs the trained agent and injects two manual 'shoves' (perturbing theta_dot directly)
    so the recovery is visible, not just the initial swing-up."""
    model = tf.keras.models.load_model(model_path, compile=False)
    env = gym.make("Pendulum-v0")
    env.seed(seed)
    env.unwrapped.g = gravity
    state = env.reset()
    thetas, rewards, torques, disturbed = [], [], [], []
    rng = np.random.default_rng(seed)
    for step in range(max_steps):
        if step in disturbance_steps:
            th, thdot = env.unwrapped.state
            kick = rng.choice([-1, 1]) * rng.uniform(2.5, 3.5)
            env.unwrapped.state = np.array([th, thdot + kick])
            state = env.unwrapped._get_obs()
        theta = np.arctan2(state[1], state[0])
        thetas.append(theta)
        disturbed.append(step in disturbance_steps)
        q_vals = model(state[None, :].astype(np.float32), training=False).numpy()[0]
        a_idx = int(np.argmax(q_vals))
        torques.append(ACTION_BINS[a_idx])
        state, reward, done, info = env.step(np.array([ACTION_BINS[a_idx]]))
        rewards.append(reward)
    env.close()
    return np.array(thetas), np.array(rewards), np.array(torques), np.array(disturbed)


thetas, rewards, torques, disturbed = rollout_with_disturbance(
    "best_dqn_pendulum.h5", gravity=10.0, seed=3)
print(f"Rollout: {len(thetas)} steps, total reward={rewards.sum():.1f}, "
      f"{disturbed.sum()} disturbance steps at {np.where(disturbed)[0].tolist()}")

# ---------- Attempt 1: live popup window (gym's own renderer) ----------
# Only works with a real display attached (a local machine, not Colab / a headless server).
try:
    demo_env = gym.make("Pendulum-v0")
    demo_env.seed(3)
    demo_env.unwrapped.g = 10.0
    demo_env.reset()
    for step in range(len(thetas)):
        demo_env.unwrapped.state = np.array([thetas[step], 0.0])  # replay the exact trajectory
        demo_env.render(mode="human")
    demo_env.close()
    print("Live popup window rendered successfully (gym's own renderer).")
except Exception as e:
    print(f"Live popup unavailable in this environment ({type(e).__name__}).")
    print("Expected on Colab / headless servers -- falling back to an embedded video below.")
    print("Running locally with a display? Re-run this cell to get the popup instead.")

# ---------- Attempt 2 (fallback, always shown): faster embedded video with a torque-direction arrow ----------
FRAME_STRIDE = 2  # play every 2nd simulation step -> ~2x faster without cutting the story short
frame_idx = np.arange(0, len(thetas), FRAME_STRIDE)

fig, (ax_pend, ax_reward) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.3]})
ax_pend.set_xlim(-1.3, 1.3); ax_pend.set_ylim(-1.3, 1.3)
ax_pend.set_aspect("equal"); ax_pend.axis("off")
ax_pend.set_title("Trained DQN agent — default gravity (g=10)")
rod, = ax_pend.plot([], [], lw=4, color="#1E2761", solid_capstyle="round")
bob = ax_pend.scatter([], [], s=500, color="#3B82C4", zorder=5, edgecolors="#1E2761", linewidths=1.5)
ax_pend.scatter([0], [0], s=80, color="#1A1A2E", zorder=6)
ax_pend.plot([0, 0], [0, 1.15], linestyle="--", color="#B3261E", alpha=0.4, lw=1.5)
step_text = ax_pend.text(-1.25, 1.15, "", fontsize=10)
kick_text = ax_pend.text(-1.25, -1.2, "", fontsize=12, color="#D97706", weight="bold")

# torque-direction arrow: curves around the pivot the way the agent is turning it, sized by strength
torque_arrow = mpatches.FancyArrowPatch((0, 0), (0, 0), connectionstyle="arc3,rad=0.3",
                                         arrowstyle="-|>", mutation_scale=15, color="#16A34A", linewidth=2.5)
ax_pend.add_patch(torque_arrow)

ax_reward.set_xlim(0, len(rewards)); ax_reward.set_ylim(min(rewards.min(), -17) - 1, 1)
ax_reward.set_xlabel("step"); ax_reward.set_ylabel("instantaneous reward")
ax_reward.set_title(f"Reward trace (total = {rewards.sum():.0f})")
ax_reward.grid(alpha=0.3)
for d in np.where(disturbed)[0]:
    ax_reward.axvline(d, color="#D97706", alpha=0.3, lw=1)
reward_line, = ax_reward.plot([], [], color="#3B82C4", lw=1.5)
reward_dot = ax_reward.scatter([], [], color="#B3261E", zorder=5, s=40)
plt.tight_layout()

def update(i):
    frame = frame_idx[i]
    theta = thetas[frame]
    x, y = np.sin(theta), np.cos(theta)
    rod.set_data([0, x], [0, y])
    bob.set_offsets([[x, y]])
    step_text.set_text(f"step {frame+1}/{len(thetas)}  torque={torques[frame]:+.2f} N·m")

    # arrow curves clockwise for negative torque, counterclockwise for positive; size = |torque|
    torque_frac = torques[frame] / 2.0
    if abs(torque_frac) > 0.05:
        r = 0.35
        spread = 0.5 + 0.4 * abs(torque_frac)
        a1 = theta - np.sign(torque_frac) * spread / 2
        a2 = theta + np.sign(torque_frac) * spread / 2
        p1 = (np.sin(a1) * (1 + r), np.cos(a1) * (1 + r))
        p2 = (np.sin(a2) * (1 + r), np.cos(a2) * (1 + r))
        torque_arrow.set_positions(p1, p2)
        torque_arrow.set_connectionstyle(f"arc3,rad={0.4 * np.sign(torque_frac)}")
        torque_arrow.set_alpha(min(1.0, abs(torque_frac) + 0.3))
        torque_arrow.set_linewidth(1.5 + 3 * abs(torque_frac))
    else:
        torque_arrow.set_alpha(0)

    recent = disturbed[max(0, frame - 8):frame + 1].any()
    kick_text.set_text("← external shove! →" if recent else "")
    bob.set_color("#D97706" if recent else "#3B82C4")
    reward_line.set_data(np.arange(frame + 1), rewards[:frame + 1])
    reward_dot.set_offsets([[frame, rewards[frame]]])
    return rod, bob, reward_line, reward_dot, torque_arrow

ani = animation.FuncAnimation(fig, update, frames=len(frame_idx), blit=False, interval=1000/30)
HTML(ani.to_html5_video())  # embeds a real playable MP4 inline -- this is the cell's output

# %% [markdown]
# Two things worth pointing out if you're presenting this: the agent was **never trained with disturbances** — it only ever saw the standard reset-and-balance task during training (Sections 5-6) — so recovering from an unexpected shove is a genuine generalisation test, not something it memorised. And the recovery isn't instantaneous: watch the reward trace dip sharply right after each shove (the pendulum swings most of the way back around before catching itself), which is a more honest and more interesting demonstration than a pendulum that simply never gets perturbed in the first place. If you have a local display, re-running the cell above will open the live gym popup instead of relying on the fallback video.

# %% [markdown]
# ## 11. Summary & Conclusions
#
# **What we built.** A DQN agent (64-64 MLP, 9 discretised torque actions, experience replay + target network per the Topic 6 lecture's two key DQN ingredients) that learns to balance `Pendulum-v0`, trained and evaluated across 4 gravity regimes with 2 seeds each (8 runs, 180 episodes/run).
#
# **What we found.**
# - The agent successfully learns a substantially-better-than-random policy in **all four** gravity settings — even the hardest setting (supergravity) improves from roughly -1500 (early, mostly-random) to roughly -321 (converged) average episode reward.
# - Gravity's *sign* relative to the upright equilibrium — whether it destabilises or self-stabilises the goal state — is the dominant factor in difficulty, far more than its magnitude. Free-fall and anti-gravity (both non-destabilising at upright) converge to policies 5-7x better, 6-8x more stable, and 4-8x faster than default and supergravity (both destabilising).
# - By our explicit "final performance, then stability, then speed" criteria, **free-fall (g=0)** is the best of the four setups; anti-gravity is a close runner-up that would win under a "fastest learning" priority instead.
#
# **Limitations & what we'd extend with more compute.** This notebook was built and fully executed on a single CPU core with no GPU, which shaped several choices documented along the way: a small 64-64 network, 9 action bins rather than more, training every 4 steps rather than every step, 180 episodes, and only 2 seeds per setting. For the final submission, the recommended extensions (all of which just mean raising a constant, not changing the code) are:
# - **More seeds** (3-5 instead of 2) for tighter confidence intervals on the Section 7 table, especially for the noisier default/supergravity settings.
# - **More episodes**, particularly for default and supergravity — their learning curves in Section 6 are still trending upward at episode 180, so it's an open question whether they'd eventually close more of the gap to the easy pair given more training time.
# - **Optional architecture extensions**, only after the DQN baseline above is solid (per the assignment brief) — Double DQN (decoupling action-selection from action-evaluation to reduce Q-value overestimation) would be a natural next step given how noisy the default/supergravity Q-estimates likely are, and ties in well with the Part C literature-review topics if RL is chosen there.
#
# **Model improvement, concretely (Section 8).** Rather than only citing the Topic 6 lecture's claims about why the target network and replay buffer matter, we ran a controlled ablation: same hyperparameters, same gravity setting, one ingredient removed at a time. Both ablated variants (no target network: -1225 mean reward; no replay buffer: -1485) essentially failed to learn within the same budget the full agent (-288) used to reach a working policy — turning a lecture claim into a verified, quantified finding.
#
# **Deliverables produced:** `best_dqn_pendulum.h5` (best overall config) plus one weights file per gravity setting in `all_gravity_weights/`, saved for reproducibility as required by the assignment brief. Section 10 additionally provides a presentable, embedded video demo of the trained agent recovering from live disturbances, useful for the presentation/demo component of this assignment.
