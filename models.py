"""
Data models for the Layout Environment.

The layout environment is an RL environment for training language models
to iteratively improve UI poster layouts via discrete actions.
"""

from typing import Dict, List, Optional

from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


ACTIONS = {
    "MOVE": ["UP", "DOWN", "LEFT", "RIGHT"],
    "RESIZE": ["WIDER", "NARROWER", "TALLER", "SHORTER"],
    "ALIGN": ["LEFT", "CENTER_X", "RIGHT", "TOP", "CENTER_Y", "BOTTOM"],
    "SNAP": ["GRID"],
    "NO_OP": ["NONE"],
}

MAGNITUDES = {
    "SMALL": 0.01,
    "MEDIUM": 0.05,
    "LARGE": 0.1,
}

ALL_VALID_PARAMS = {param for params in ACTIONS.values() for param in params}


class LayoutAction(Action):
    """
    Action for the Layout environment. The agent selects one element and applies one operation per step.

    Attributes:
        element_id: Index of the target element in the layout.
        action: One of "MOVE", "RESIZE", "ALIGN", "SNAP", "NO_OP".
        param: Parameter for the action (e.g. "UP", "WIDER", "CENTER_X").
        magnitude: Step size for MOVE/RESIZE — "SMALL" (0.01), "MEDIUM" (0.05),
                   "LARGE" (0.10). Ignored for other action types.
    """

    element_id: int = Field(default=0, description="Target element index")
    action: str = Field(default="NO_OP", description="Action type")
    param: str = Field(default="NONE", description="Action parameter")
    magnitude: str = Field(default="MEDIUM", description="Step size for MOVE/RESIZE")

    def is_valid(self, num_elements: int) -> bool:
        # Validate action type
        if self.action not in ACTIONS:
            return False

        # Validate param for the given action
        valid_params = ACTIONS[self.action]
        if self.param not in valid_params:
            return False

        # Validate element index (except NO_OP)
        if self.action != "NO_OP":
            if not isinstance(self.element_id, int):
                return False
            if not (0 <= self.element_id < num_elements):
                return False

        if self.action in ["MOVE", "RESIZE"] and self.magnitude not in MAGNITUDES:
            return False

        return True

    def to_dict(self) -> Dict:
        return {
            "element_id": self.element_id,
            "action": self.action,
            "param": self.param,
            "magnitude": self.magnitude,
        }


class LayoutObservation(Observation):
    """
    Observation from the Layout environment.

    All float coordinates are in normalised [0, 1] space and rounded to 3 decimal places to minimise LM token count.

    Attributes:
        canvas: Canvas dimensions (always {"width": 1.0, "height": 1.0}).
        elements: Current element list with id, type, cx, cy, w, h, font_size.
        metrics: Per-metric scores (overlap, boundary, occlusion, alignment, spacing, plausibility).
            Content-aware metrics (e.g. occlusion) are active in VLM mode and
            neutralized in LLM mode.
        step: Current step within the episode.
        max_steps: Maximum steps allowed in this episode.
        quality_score: Composite Q(state) — higher is better.
        initial_quality_score: Q(state_0) at the start of this episode.
        image_path: In VLM mode, path relative to the dataset JSON to the background (e.g. ``images/id_bg.png``).
        rendered_image_base64: Optional in VLM mode. PNG of the background with
            layout boxes and labels, base64-encoded, when server-side rendering
            is enabled for the episode.
    """

    canvas: Dict = Field(default_factory=lambda: {"width": 1.0, "height": 1.0})
    elements: List[Dict] = Field(default_factory=list)
    metrics: Dict = Field(default_factory=dict)

    step: int = 0
    max_steps: int = 500

    quality_score: float = 0.0
    initial_quality_score: float = 0.0

    text_feedback: Optional[str] = None
    reward: float = 0.0
    done: bool = False

    image_path: Optional[str] = None          # background image path
    rendered_image_base64: Optional[str] = None  # layout rendered on background (visual prompt)
    
class LayoutState(State):
    """
    State of the Layout environment.
    """
    # Base State provides: episode_id, step_count
    elements: List[Dict] = Field(default_factory=list)
    
    # Quality tracking (for delta calculation)
    previous_quality: float = 0.0  # Q(t-1)
    initial_quality: float = 0.0   # Q(0)
    
    current_image_rel: Optional[str] = None
    current_saliency_rel: Optional[str] = None
    dataset_json_path: Optional[str] = None