---
title: Cosmic Bytes Robot Task Sequencer
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Cosmic Bytes: Robot Task Sequencer

**Cosmic Bytes** is an RL environment where an AI agent must produce the correct ordered sequence of robot actions to complete specific tasks based on vision-language inputs. It serves as a benchmark for evaluating the planning and sequencing capabilities of large multimodal models and RL agents in robotics-inspired domains.

## Motivation

As AI moves closer to physical action, the ability to plan long-term sequences of discrete actions based on visual scene understanding becomes critical. Cosmic Bytes provides a structured way to test if models can:
1.  Parse a visual scene for relevant objects.
2.  Decompose a natural language goal into atomic steps.
3.  Adhere to physical constraints (e.g., clearing hazards before retrieval).
4.  Handle progressive difficulty through multi-stage dependency tasks.

---

## Environment Design

### Observation Space
The observation space is designed to provide all necessary context for the agent:
- `task_id` (str): ID of the current active session.
- `task_description` (str): Natural language description of the goal.
- `image_base64` (Optional[str]): A base64-encoded RGB image of the workspace.
- `available_actions` (List[str]): A list of valid strings representing permitted robot actions.
- `attempt` (int): The current attempt number in the episode.
- `max_attempts` (int): Maximum attempts allowed before the episode terminates.
- `hint` (Optional[str]): A context-aware hint provided after failed attempts.

### Action Space
The agent produces a single, structured action per step:
- `step_sequence` (List[str]): An ordered list of action strings chosen from the `available_actions` list.

### Reward Function
The environment uses a **Longest Common Subsequence (LCS)** grader to provide a dense reward signal (0.0 – 1.0):
- **1.0**: Perfect sequence match.
- **Partial Credit**: Based on the ratio of correct steps in the correct order relative to the ground truth.
- **Penalty**: A small penalty is applied for extra, unnecessary steps to encourage efficiency.

---

## Tasks & Difficulty

| ID | Difficulty | Goal | Key Challenge |
| :--- | :--- | :--- | :--- |
| `easy_sorting` | Easy | Sort color-coded blocks into matching bins. | Basic spatial mapping and classification. |
| `medium_assembly` | Medium | Assemble a 4-part robot chassis in order. | Structural dependency (bottom-up assembly). |
| `hard_multistep` | Hard | Retrieve an item from a cluttered spill zone. | Hazard clearance and prerequisite management. |

---

## Setup & Usage

### 🚀 Deployment to HF Spaces
From the root directory:
```bash
openenv push
```

### 🐳 Running Locally with Docker
1. **Build**:
   ```bash
   docker build -t cosmic-bytes:latest .
   ```
2. **Run**:
   ```bash
   docker run -p 7860:7860 cosmic-bytes:latest
   ```

   ```bash
   docker run -p 7860:7860 --env-file .env cosmic-bytes:latest
   ```

   ```bash
   uv run python inference.py
   ```

### 🐍 Python Usage
```python
from cosmic_bytes import CosmicBytesEnv

with CosmicBytesEnv(base_url="http://localhost:7860") as env:
    obs = env.reset("easy_sorting")
    result = env.step(CosmicBytesAction(step_sequence=["pick_up_red_block", "move_to_red_bin", "place_in_red_bin"]))
    print(f"Goal: {obs.task_description}")
    print(f"Outcome: {result.observation.reward} reward")
```

---

## Baseline Performance

Results using `meta-llama/Llama-3.3-70B-Instruct` as a zero-shot planner:

| Task | Success | Average Score |
| :--- | :--- | :--- |
| Easy Sorting | **✓** | 1.000 |
| Medium Assembly | **✓** | 1.000 |
| Hard Multistep | **✓** | 1.000 |
| **Total Average** | - | **1.000** |

---

## Compliance & Ethics
This environment is designed for research purposes and does not model real-world hazardous conditions with physical fidelity. It adheres strictly to the OpenEnv specification v0.2.2.
