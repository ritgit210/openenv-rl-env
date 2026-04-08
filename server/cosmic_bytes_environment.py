"""
server/cosmic_bytes_environment.py
Full stateful logic for all task levels.
"""

import base64
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import CosmicBytesAction, CosmicBytesObservation
except ImportError:
    from models import CosmicBytesAction, CosmicBytesObservation

# ── Task Definitions ─────────────────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    "easy_sorting": {
        "difficulty": "easy",
        "objects": ["red_block", "blue_block", "green_block"],
        "goals": ["red_block_sorted", "blue_block_sorted", "green_block_sorted"],
        "available_actions": [
            "pick_up_red_block", "pick_up_blue_block", "pick_up_green_block",
            "move_to_red_bin", "move_to_blue_bin", "move_to_green_bin",
            "place_in_red_bin", "place_in_blue_bin", "place_in_green_bin"
        ],
        "description": "Sort the colored blocks into matching bins.",
    },
    "medium_assembly": {
        "difficulty": "medium",
        "objects": ["base", "panel", "cover"],
        "goals": ["base_placed", "side_panel_1", "side_panel_2", "cover_attached", "assembly_secured"],
        "available_actions": [
            "pick_up_base", "place_base", "pick_up_panel", "attach_panel", 
            "pick_up_cover", "attach_cover", "tighten_screws"
        ],
        "description": "Assemble the chassis in order: Base -> 2 Panels -> Cover -> Screws.",
    },
    "hard_multistep": {
        "difficulty": "hard",
        "objects": ["spill", "flask", "locked_box", "item"],
        "goals": ["hazards_assessed", "spill_cleaned", "flask_cleared", "box_unlocked", "box_opened", "item_retrieved"],
        "available_actions": [
            "assess_workspace", "clean_spill", "pick_up_flask", "place_flask_safely",
            "unlock_box", "open_box", "pick_up_item", "place_in_shipping_crate"
        ],
        "description": "Clear all hazards (spill/flask) before opening the box and retrieving the item.",
    }
}

# ── Image Caching ────────────────────────────────────────────────────────────
_IMAGE_CACHE: Dict[str, str] = {}

def preload_images():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    if not data_dir.exists():
        return
    for t_id in TASKS.keys():
        p = data_dir / f"{t_id}.png"
        if p.exists():
            _IMAGE_CACHE[t_id] = base64.b64encode(p.read_bytes()).decode("utf-8")

preload_images()

class CosmicBytesEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "easy_sorting"):
        self._task_id = task_id
        if self._task_id not in TASKS:
            # Safe fallback
            self._task_id = "easy_sorting"
        self._task = TASKS[self._task_id]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        # Physical State tracking
        self._inventory: Optional[str] = None
        self._at_location: str = "table"
        self._completed_goals: List[str] = []
        self._done = False
        self._max_steps = 15
        self._current_cumulative_reward = 0.05 # Initial baseline rounds to 0.05

    def reset(self) -> CosmicBytesObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._inventory = None
        self._at_location = "table"
        self._completed_goals = []
        self._done = False
        self._current_cumulative_reward = 0.05
        return self._build_observation("Environment Reset.")

    def step(self, action: CosmicBytesAction) -> CosmicBytesObservation: # type: ignore[override]
        if self._done:
            return self._build_observation("Episode finished.")

        total_reward = 0.0
        feedbacks = []
        
        # Handle both list and string sequences based on CosmicBytesAction model
        sequence = action.step_sequence
        if isinstance(sequence, str):
            sequence = [sequence]
            
        for input_action in sequence:
            if self._done:
                break
                
            self._state.step_count += 1
            reward, feedback = self._process_logic(input_action)
            total_reward += reward
            feedbacks.append(f"[{input_action}] {feedback}")
            
            # Check for completion or failure after each sub-step
            if len(self._completed_goals) == len(self._task.get("goals", [])):
                self._done = True
                feedbacks.append("[TASK SUCCESS]")
                break
            elif self._state.step_count >= self._max_steps:
                self._done = True
                feedbacks.append("[TIMEOUT]")
                break
            
        # Calculate new cumulative score based on progress: strictly in (0.05, 0.95)
        # We use increments of 0.01 to ensure they are visible after 2-decimal rounding.
        goals = self._task.get("goals", [])
        total_goals = len(goals) if goals else 1
        completion_ratio = min(1.0, len(self._completed_goals) / total_goals)
        
        # target_cumulative = base(0.1) + progress(0.7) + step_bonus(0.01*step)
        # Max steps = 15, so max target = 0.1 + 0.7 + 0.15 = 0.95
        # Min target (step 1) = 0.1 + 0 + 0.01 = 0.11
        target_cumulative = 0.10 + round(0.70 * completion_ratio, 2) + round(0.01 * self._state.step_count, 2)
        
        # Reward this step is difference to reach target, at least 0.01
        final_reward = round(max(0.01, target_cumulative - self._current_cumulative_reward), 2)
        self._current_cumulative_reward = round(self._current_cumulative_reward + final_reward, 2)

        obs = self._build_observation(" | ".join(feedbacks) if feedbacks else "No action sequence provided.")
        obs.reward = float(final_reward)
        obs.done = self._done
        return obs

    def _process_logic(self, action: str) -> tuple[float, str]:
        reward = -0.05
        
        # 1. TASK SPECIFIC OVERRIDES (Checked First)
        if self._task_id == "easy_sorting":
            if action.startswith("place_in_"):
                target = action.replace("place_in_", "")
                if not self._inventory: return -0.3, "Failed: Empty gripper."
                color = self._inventory.split("_")[0] if "_" in str(self._inventory) else str(self._inventory)
                if target == f"{color}_bin" and self._at_location == f"{color}_bin":
                    goal = f"{self._inventory}_sorted"
                    if goal not in self._completed_goals:
                        self._completed_goals.append(goal)
                        self._inventory = None
                        return 0.5, f"Sorted {color} block."
                    else:
                        self._inventory = None
                        return 0.0, f"{color} block already sorted."
                return -0.2, "Failed: Wrong bin or location."

        elif self._task_id == "medium_assembly":
            if action == "place_base":
                if self._inventory == "base":
                    goal = "base_placed"
                    if goal not in self._completed_goals:
                        self._completed_goals.append(goal)
                        self._inventory = None
                        return 0.5, "Base plate secured."
                    return 0.0, "Base already placed."
                return -0.2, "Failed: Not holding base."
            elif action == "attach_panel":
                if "base_placed" not in self._completed_goals: return -0.3, "Failed: No foundation."
                if self._inventory == "panel":
                    if "side_panel_1" not in self._completed_goals: goal = "side_panel_1"
                    elif "side_panel_2" not in self._completed_goals: goal = "side_panel_2"
                    else: return 0.0, "All panels attached."
                    self._completed_goals.append(goal)
                    self._inventory = None
                    return 0.5, f"Attached {goal}."
            elif action == "attach_cover":
                if "side_panel_2" not in self._completed_goals: return -0.3, "Failed: Panels missing."
                if self._inventory == "cover":
                    goal = "cover_attached"
                    if goal not in self._completed_goals:
                        self._completed_goals.append(goal)
                        self._inventory = None
                        return 0.5, "Cover attached."
                    return 0.0, "Cover already attached."
            elif action == "tighten_screws":
                if "cover_attached" not in self._completed_goals: return -0.3, "Failed: Cover not on."
                goal = "assembly_secured"
                if goal not in self._completed_goals:
                    self._completed_goals.append(goal)
                    return 0.5, "Assembly secured."
                return 0.0, "Screws already tight."

        elif self._task_id == "hard_multistep":
            if action == "assess_workspace":
                goal = "hazards_assessed"
                if goal not in self._completed_goals:
                    self._completed_goals.append(goal)
                    return 0.2, "Workspace scanned. Spill detected south."
                return 0.0, "Already assessed."
            elif action == "clean_spill":
                if "hazards_assessed" not in self._completed_goals: return -0.2, "Error: Assess first."
                goal = "spill_cleaned"
                if goal not in self._completed_goals: 
                    self._completed_goals.append(goal)
                    return 0.5, "Spill neutralized."
                return 0.0, "Spill already cleaned."
            elif action == "place_flask_safely":
                if "spill_cleaned" not in self._completed_goals: return -0.3, "Safety Violation: Cleanup first."
                if self._inventory == "flask":
                    goal = "flask_cleared"
                    if goal not in self._completed_goals: 
                        self._completed_goals.append(goal)
                        self._inventory = None
                        return 0.4, "Flask cleared."
                    self._inventory = None
                    return 0.0, "Flask already cleared."
            elif action == "unlock_box":
                if "flask_cleared" not in self._completed_goals: return -0.3, "Failed: Hazards still blocking pass."
                goal = "box_unlocked"
                if goal not in self._completed_goals:
                    self._completed_goals.append(goal)
                    return 0.4, "Box unlocked."
                return 0.0, "Already unlocked."
            elif action == "open_box":
                if "box_unlocked" not in self._completed_goals: return -0.2, "Locked: Box won't budge."
                goal = "box_opened"
                if goal not in self._completed_goals: 
                    self._completed_goals.append(goal)
                    return 0.5, "Box opened. Target item visible."
                return 0.0, "Box already open."
            elif action == "pick_up_item":
                if "box_opened" not in self._completed_goals: return -0.3, "Failed: The item is still locked inside the box."
                if self._inventory: return -0.2, f"Failed: Already holding {self._inventory}."
                self._inventory = "item"
                return 0.2, "Picked up the target item."
            elif action == "place_in_shipping_crate":
                if self._inventory == "item" and "box_opened" in self._completed_goals:
                    goal = "item_retrieved"
                    if goal not in self._completed_goals: 
                        self._completed_goals.append(goal)
                        self._inventory = None
                        return 0.8, "Item secured in crate."
                    self._inventory = None
                    return 0.0, "Item already retrieved."

        # 2. UNIVERSAL ACTIONS (Fallback)
        if action.startswith("pick_up_"):
            obj = action.replace("pick_up_", "")
            if self._inventory: return -0.2, f"Failed: Already holding {self._inventory}."
            self._inventory = obj
            return 0.2, f"Picked up {obj}."

        elif action.startswith("move_to_"):
            loc = action.replace("move_to_", "")
            self._at_location = loc
            return 0.1, f"Moved to {loc}."

        return reward, f"Action '{action}' had no effect."

        return reward, f"Action '{action}' had no effect."

    @property
    def state(self) -> State:
        return self._state

    def _build_observation(self, feedback: str) -> CosmicBytesObservation:
        state_desc = f"At: {self._at_location} | Holding: {self._inventory or 'None'} | Goals: {', '.join(self._completed_goals) if self._completed_goals else 'None'}"
        # Task score is derived from cumulative reward, rounded to 2 decimal places
        current_score = round(max(0.01, min(0.99, self._current_cumulative_reward)), 2)
        
        return CosmicBytesObservation(
            task_id=self._task_id,
            task_description=f"{self._task['description']} | State: {state_desc} | Last Result: {feedback}",
            image_base64=self._load_image(),
            available_actions=self._task["available_actions"],
            attempt=self._state.step_count,
            max_attempts=self._max_steps,
            done=self._done,
            reward=0.01, # Default non-zero reward
            score=current_score
        )

    def _load_image(self) -> Optional[str]:
        return _IMAGE_CACHE.get(self._task_id)
