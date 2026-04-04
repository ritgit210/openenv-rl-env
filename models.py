from typing import Optional, List
from pydantic import Field
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


class CosmicBytesAction(Action):
    """Action for the Robot Task Sequencer environment."""

    step_sequence: List[str] = Field(..., description="Ordered list of action strings")


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