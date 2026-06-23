"""Command-line entry point: `python -m app`."""
from __future__ import annotations

import argparse

import uvicorn

from .main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app",
        description="Run the llm-api-proxy server.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Disable use of the system HTTP/HTTPS proxy when calling upstream providers.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8089, help="Bind port (default: 8089)")
    args = parser.parse_args()

    app = create_app(no_proxy=args.no_proxy)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
