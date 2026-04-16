"""
Layout Environment Client.

HTTP / WebSocket client for interacting with a remote LayoutEnvironment server.
"""

import os
import urllib.error
import urllib.request
from typing import Dict


from openenv.core.client_types import StepResult  # noqa: F401
from openenv.core.env_client import EnvClient

try:
    from .models import LayoutAction, LayoutObservation, LayoutState
except ImportError:
    from models import LayoutAction, LayoutObservation, LayoutState


def layout_env_kwargs_from_environ() -> Dict[str, float]:
    """
    Extra kwargs for :class:`LayoutEnv` (OpenEnv ``EnvClient``).

    Default WebSocket connect timeout in openenv is 10s, which is often too low for
    Hugging Face Spaces that are cold-starting or waking from sleep.
    """
    return {
        "connect_timeout_s": float(os.getenv("LAYOUTENV_WS_CONNECT_TIMEOUT", "120")),
        "message_timeout_s": float(os.getenv("LAYOUTENV_WS_MESSAGE_TIMEOUT", "120")),
    }


def warmup_hf_space_http(base_url: str, timeout_s: float = 180.0) -> None:
    """
    Best-effort HTTP GET to the Space origin so a sleeping replica can boot
    before the WebSocket handshake (reduces ``timed out during opening handshake``).
    """
    if os.getenv("LAYOUTENV_SKIP_SPACE_WARMUP", "").lower() in ("1", "true", "yes"):
        return
    base = base_url.strip().rstrip("/")
    if not base.startswith("http"):
        base = "https://" + base
    if ".hf.space" not in base.lower():
        return
    req = urllib.request.Request(base, headers={"User-Agent": "layoutenv-client/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            resp.read(65536)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass


class LayoutEnv(EnvClient[LayoutAction, LayoutObservation, LayoutState]):
    """
    Client for the Layout Environment.

    Example:
        >>> with LayoutEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.quality_score)
        ...     result = client.step(LayoutAction(
        ...         element_id=0, action="MOVE", param="UP", magnitude="LARGE",
        ...     ))
    """

    def _step_payload(self, action: LayoutAction) -> Dict:
        return action.to_dict()

    def _parse_result(self, payload: Dict) -> StepResult[LayoutObservation]:
        if "observation" not in payload:
            raise ValueError(f"Invalid response: {payload}")

        obs_data = payload["observation"]

        observation = LayoutObservation(
            canvas=obs_data.get("canvas", {"width": 1.0, "height": 1.0}),
            elements=obs_data.get("elements", []),
            metrics=obs_data.get("metrics", {}),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 20),
            quality_score=obs_data.get("quality_score", 0.0),
            initial_quality_score=obs_data.get("initial_quality_score", 0.0),
            text_feedback=obs_data.get("text_feedback"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
            image_path=obs_data.get("image_path"),
            rendered_image_base64=obs_data.get("rendered_image_base64"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> LayoutState:
        return LayoutState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
