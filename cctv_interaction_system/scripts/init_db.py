"""Initialise the PostgreSQL schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.db_client import PostgresClient  # noqa
from src.common.logger import get_logger, setup_logger  # noqa
from config.settings import get_settings  # noqa

setup_logger()
logger = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--database", default=None)
    args = parser.parse_args()

    cfg = get_settings().postgres
    client = PostgresClient(
        host=args.host or cfg.host,
        port=args.port or cfg.port,
        user=args.user or cfg.user,
        password=args.password or cfg.password,
        database=args.database or cfg.database,
    )
    try:
        client.init_schema()
        logger.info("schema initialised")
    except Exception as e:
        logger.error(f"failed: {e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
