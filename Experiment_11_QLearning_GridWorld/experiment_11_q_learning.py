"""
Experiment 11: Q-Learning in a Grid World Environment
Evaluates reinforcement learning for decision-making, exploration vs. exploitation, and optimal policy.
"""

import numpy as np
import pandas as pd

def main():
    print("=" * 70)
    print("EXPERIMENT 11: REINFORCEMENT LEARNING USING Q-LEARNING (GRID WORLD)")
    print("=" * 70)

    np.random.seed(42)

    # 1. Environment parameters
    GRID_SIZE = 4
    START = (0, 0)
    GOAL = (3, 3)
    WALLS = {(1, 1), (2, 1)}
    ACTIONS = ["U", "R", "D", "L"]
    ACTION_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    valid_cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if (r, c) not in WALLS]
    state_to_pos = {i: pos for i, pos in enumerate(valid_cells)}
    pos_to_state = {pos: i for i, pos in state_to_pos.items()}
    START_STATE = pos_to_state[START]
    GOAL_STATE = pos_to_state[GOAL]
    N_STATES, N_ACTIONS = len(valid_cells), len(ACTIONS)

    def reset():
        return START_STATE

    def step(state, action):
        r, c = state_to_pos[state]
        dr, dc = ACTION_DELTAS[action]
        candidate = (r + dr, c + dc)
        if (candidate[0] < 0 or candidate[0] >= GRID_SIZE or
            candidate[1] < 0 or candidate[1] >= GRID_SIZE or
            candidate in WALLS):
            return state, -5.0, False
        next_state = pos_to_state[candidate]
        if next_state == GOAL_STATE:
            return next_state, 10.0, True
        return next_state, -1.0, False

    # 2. Q-Learning training
    alpha = 0.1
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995
    episodes = 2500
    max_steps = 100

    Q = np.zeros((N_STATES, N_ACTIONS), dtype=float)
    episode_rewards = []
    episode_steps = []
    episode_success = []

    def choose_action(state, eps):
        if np.random.random() < eps:
            return np.random.randint(N_ACTIONS)
        values = Q[state]
        best_actions = np.flatnonzero(np.isclose(values, values.max()))
        return int(np.random.choice(best_actions))

    for episode in range(episodes):
        state = reset()
        total_reward = 0.0
        success = False
        for t in range(max_steps):
            action = choose_action(state, epsilon)
            next_state, reward, done = step(state, action)
            future_val = 0.0 if done else np.max(Q[next_state])
            target = reward + gamma * future_val
            Q[state, action] += alpha * (target - Q[state, action])
            total_reward += reward
            state = next_state
            if done:
                success = True
                break
        episode_rewards.append(total_reward)
        episode_steps.append(t + 1)
        episode_success.append(int(success))
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Training completed across {episodes} episodes.")
    print(f"Final epsilon: {epsilon:.3f}")
    print(f"Last 100-episode success rate: {np.mean(episode_success[-100:]) * 100:.1f}%")

    # 3. Greedy Policy Evaluation
    successes = 0
    eval_rewards = []
    eval_steps = []
    sample_path = []

    for i in range(200):
        state = reset()
        tot_r = 0.0
        path = [state]
        done = False
        for t in range(max_steps):
            best = np.flatnonzero(np.isclose(Q[state], Q[state].max()))
            action = int(np.random.choice(best))
            next_state, reward, done = step(state, action)
            tot_r += reward
            state = next_state
            path.append(state)
            if done:
                break
        successes += int(done)
        eval_rewards.append(tot_r)
        eval_steps.append(len(path) - 1)
        if i == 0:
            sample_path = path

    print(f"\nEvaluation over 200 greedy episodes:")
    print(f"  Success Rate : {successes / 200 * 100:.1f}%")
    print(f"  Mean Reward  : {np.mean(eval_rewards):.2f}")
    print(f"  Mean Steps   : {np.mean(eval_steps):.2f}")
    print(f"  Sample Path  : {[state_to_pos[s] for s in sample_path]}")

    # 4. Learned Q-Table
    q_df = pd.DataFrame(
        Q.round(2),
        columns=[f"Action {a}" for a in ACTIONS],
        index=[f"State {s} {state_to_pos[s]}" for s in range(N_STATES)]
    )
    print("\nLearned Q-Table:")
    print(q_df.to_string())

    # 5. Extract Policy (ASCII friendly for terminal output)
    ARROWS = {"U": "UP (^)", "R": "RIGHT (>)", "D": "DOWN (v)", "L": "LEFT (<)"}
    print("\nLearned Greedy Directions:")
    for s in range(N_STATES):
        pos = state_to_pos[s]
        if s == GOAL_STATE:
            dir_str = "GOAL (G)"
        else:
            best = np.flatnonzero(np.isclose(Q[s], Q[s].max()))
            dir_str = ARROWS[ACTIONS[int(best[0])]]
        print(f"  Pos {pos}: {dir_str}")

    print("\nExperiment 11 executed successfully!")

if __name__ == "__main__":
    main()
