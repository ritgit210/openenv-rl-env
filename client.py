"""
client.py — Python client for the Cosmic Bytes OpenEnv server.
"""

from typing import Dict, Any, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CosmicBytesAction, CosmicBytesObservation
except ImportError:
    from models import CosmicBytesAction, CosmicBytesObservation


class CosmicBytesEnv(
    EnvClient[CosmicBytesAction, CosmicBytesObservation, State]
):
    """
    Client for the Cosmic Bytes Environment.
    """

    def _step_payload(self, action: CosmicBytesAction) -> Dict:
        """
        Convert Action to JSON payload for step message.
        """
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[CosmicBytesObservation]:
        """
        Parse server response into StepResult.
        """
        obs_data = payload.get("observation", {})
        observation = CosmicBytesObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )