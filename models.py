from typing import Optional, List, Union, Any
import json
from pydantic import Field, field_validator
from openenv.core.env_server.types import Action, Observation


class CosmicBytesObservation(Observation):
    """Observation from the Robot Task Sequencer environment."""

    task_id: str = Field(..., description="ID of the current task")
    task_description: str = Field(..., description="Description of the goal")
    image_base64: Optional[str] = Field(None, description="base64-encoded PNG/JPG of the scene")
    available_actions: List[str] = Field(default_factory=list, description="List of valid action strings")
    attempt: int = Field(default=0, description="Current attempt number")
    max_attempts: int = Field(default=0, description="Maximum attempts allowed")
    hint: Optional[str] = Field(None, description="Optional hint for the task")
    score: float = Field(0.0, description="Current task score (0.0-1.0)")


class CosmicBytesAction(Action):
    """Action for the Robot Task Sequencer environment."""

    step_sequence: Union[List[str], str] = Field(default_factory=list, description="Ordered list of action strings or a single action string")

    @field_validator("step_sequence", mode="before")
    @classmethod
    def parse_step_sequence(cls, v: Any) -> List[str]:
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v_str: str = v.strip()
            # Try to parse as JSON list if it looks like one
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    parsed = json.loads(v_str)
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed]
                except Exception:
                    # Fallback to manual split if JSON fails (e.g. trailing comma)
                    v_str = v_str[1:-1].strip()
            
            # Manual split by comma, then clean up each element
            if "," in v_str:
                items = []
                for x in v_str.split(","):
                    # Strip whitespace and common boarders/quotes
                    item = x.strip().strip('"').strip("'").strip()
                    if item:
                        items.append(item)
                return items
            
            # Single action, clean it too
            final_v = v_str.strip('"').strip("'").strip()
            return [final_v] if final_v else []
        return []


class CosmicBytesReward(Observation):
    """Detailed reward information (sent as metadata or part of StepResult)."""
    score: float = Field(0.0, description="0.0 - 1.0")
    feedback: str = Field("", description="Natural language feedback")
    correct: bool = Field(False, description="Whether the sequence is perfect")
    steps_correct: int = Field(0, description="Number of correctly placed steps")
    steps_total: int = Field(0, description="Total steps in the ground truth")


class TaskInfo(Observation):
    """Information about a registered task."""
    task_id: str
    difficulty: str
    description: str
    available_actions: List[str]
