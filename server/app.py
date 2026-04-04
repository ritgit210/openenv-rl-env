"""
server/app.py — FastAPI HTTP server exposing the OpenEnv API.
Exposes CosmicBytesEnvironment using openenv-core's create_app.
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    # Fallback to local import if installed as a package
    from openenv.core.env_server.http_server import create_app

try:
    from cosmic_bytes.models import CosmicBytesAction, CosmicBytesObservation
    from cosmic_bytes.server.cosmic_bytes_environment import CosmicBytesEnvironment
except ImportError:
    from models import CosmicBytesAction, CosmicBytesObservation
    from server.cosmic_bytes_environment import CosmicBytesEnvironment


# Create the app with web interface and README integration
app = create_app(
    CosmicBytesEnvironment,
    CosmicBytesAction,
    CosmicBytesObservation,
    env_name="cosmic-bytes",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)