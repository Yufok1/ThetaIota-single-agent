#!/usr/bin/env python3
"""
Server launcher for production-style runs without Docker.

Usage (example):
  python server_main.py --replica-index 1

Reads environment variables (via .env if present):
  RA_BIND_HOST, RA_PORT_BASE, RA_REPLICAS, RA_DB_PATH,
  RA_CHECKPOINT_DIR, RA_AUTH_SECRET, RA_MAX_VRAM_MB

Computes port and peers from RA_PORT_BASE and replica index, then launches
Phase4APIServer on that port with appropriate config.
"""

import argparse
import asyncio
import os
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):  # fallback no-op
        return False

from phase4_api_server import Phase4APIServer


def build_config() -> Dict[str, Any]:
    load_dotenv()  # best-effort
    bind_host = os.getenv("RA_BIND_HOST", "127.0.0.1")
    port = int(os.getenv("RA_PORT_BASE", "8081"))
    db_path = os.getenv("RA_DB_PATH", "agent_A.db")
    ckpt_dir = os.getenv("RA_CHECKPOINT_DIR", "checkpoints")
    auth_secret = os.getenv("RA_AUTH_SECRET")

    config: Dict[str, Any] = {
        'agent_id': 'agent_A',
        'db_path': db_path,
        'human_feedback_enabled': True,
        'peers': [],
        'heartbeat_interval_s': 5,
        'leader_id': 'agent_A',
        'checkpoint_dir': ckpt_dir,
    }
    # Attach auth secret into process env for the API layer to read
    if auth_secret:
        os.environ['RA_AUTH_SECRET'] = auth_secret

    return config, port


async def main_async():
    config, port = build_config()
    server = Phase4APIServer(config, port=port)
    await server.start_server()


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


