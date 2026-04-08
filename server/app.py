"""
FastAPI application for the Layout Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

The mode (llm/vlm) and text_feedback flag are set per-episode via reset().
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e: 
    raise ImportError(
        "openenv is required for the web interface. "
        "Install dependencies with 'uv sync'"
    ) from e

try:
    from ..models import LayoutAction, LayoutObservation
    from .layout_environment import LayoutEnvironment
except ImportError:
    from models import LayoutAction, LayoutObservation
    from server.layout_environment import LayoutEnvironment

app = create_app(
    LayoutEnvironment,
    LayoutAction,
    LayoutObservation,
    env_name="layoutenv",
    max_concurrent_envs=1,
)


def main() -> None:
    """Entry point for direct execution via ``uv run --project . server``."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
