"""Redis client wrapper for inter-service messaging.

Uses Redis Streams for inter-process communication between layers.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import redis

from .logger import get_logger

logger = get_logger()


class RedisClient:
    """Thin wrapper around redis-py with helpers for streams + JSON."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True,
    ):
        self.client = redis.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=decode_responses, socket_keepalive=True,
            health_check_interval=30,
        )

    # -- low-level ----------------------------------------------------
    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False

    # -- key/value (JSON) --------------------------------------------
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self.client.set(key, json.dumps(value, default=str))
        if ttl:
            self.client.expire(key, ttl)

    def get_json(self, key: str) -> Optional[Any]:
        v = self.client.get(key)
        return json.loads(v) if v else None

    # -- streams ------------------------------------------------------
    def xadd(self, stream: str, fields: Dict[str, Any], maxlen: int = 10000) -> str:
        """Add entry to a Redis stream. Auto-serializes values to JSON strings."""
        serialised = {k: json.dumps(v, default=str) for k, v in fields.items()}
        return self.client.xadd(stream, serialised, maxlen=maxlen, approximate=True)

    def xread(
        self,
        streams: Dict[str, str],
        count: int = 100,
        block_ms: Optional[int] = None,
    ) -> List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]:
        """Read from streams. Returns list of (stream_name, [(id, {field: value}), ...])."""
        return self.client.xread(streams, count=count, block=block_ms or None)

    def xread_group(
        self,
        group: str,
        consumer: str,
        streams: Dict[str, str],
        count: int = 100,
        block_ms: Optional[int] = None,
    ):
        return self.client.xreadgroup(
            group, consumer, streams, count=count, block=block_ms or None
        )

    def xgroup_create(self, stream: str, group: str, id: str = "$", mkstream: bool = True) -> None:
        try:
            self.client.xgroup_create(stream, group, id=id, mkstream=mkstream)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def xack(self, stream: str, group: str, *ids: str) -> int:
        return self.client.xack(stream, group, *ids)

    # -- pub/sub ------------------------------------------------------
    def publish(self, channel: str, message: Any) -> int:
        return self.client.publish(channel, json.dumps(message, default=str))


# Module-level singleton (lazy)
_client: Optional[RedisClient] = None


def get_redis() -> RedisClient:
    global _client
    if _client is None:
        from config.settings import get_settings
        cfg = get_settings().redis
        _client = RedisClient(cfg.host, cfg.port, cfg.db, cfg.password)
    return _client


def reset_redis() -> None:
    """Reset the singleton — used in tests."""
    global _client
    _client = None
