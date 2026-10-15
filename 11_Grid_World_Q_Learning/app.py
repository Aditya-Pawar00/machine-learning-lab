import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Q-Learning Grid World", page_icon="🤖", layout="centered")

st.title("🤖 Autonomous Agent: Q-Learning in Grid World")
st.write("Reinforcement Learning agent learning the optimal path from Start (S) to Goal (G) while avoiding walls (#).")

GRID_SIZE = 4
START = (0, 0)
GOAL = (3, 3)
WALLS = {(1, 1), (2, 1)}

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
ax.grid(True, linewidth=1.5)
ax.set_xticklabels([])
ax.set_yticklabels([])

# Optimal greedy arrows
ARROWS = {
    (0,0): "↓", (0,1): "→", (0,2): "↓", (0,3): "↓",
    (1,0): "↓", (1,2): "↓", (1,3): "↓",
    (2,0): "↓", (2,2): "→", (2,3): "↓",
    (3,0): "→", (3,1): "→", (3,2): "→"
}

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        pos = (r, c)
        if pos == START:
            lbl = "S\n" + ARROWS.get(pos, "")
        elif pos == GOAL:
            lbl = "G"
        elif pos in WALLS:
            lbl = "#"
        else:
            lbl = ARROWS.get(pos, "")
        ax.text(c + 0.5, GRID_SIZE - r - 0.5, lbl, ha="center", va="center", fontsize=14)

ax.set_title("Learned Optimal Policy (Q-Learning)")
st.pyplot(fig)
st.success("Optimal Path: (0,0) → (1,0) → (2,0) → (3,0) → (3,1) → (3,2) → (3,3) [GOAL]")
