"""Redis client wrapper for inter-service messaging.

Uses Redis connection pool with retry logic for resilience.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import redis
from redis import RedisError

from .logger import get_logger

logger = get_logger()


def _retry(fn: Callable, max_attempts: int = 3, base_delay: float = 0.1) -> Any:
    """Retry a Redis call with exponential backoff."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except RedisError as e:
            last_err = e
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_err


class RedisClient:
    """Thin wrapper around redis-py with connection pool, retry, and pub/sub."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True,
        max_connections: int = 10,
    ):
        self._pool = redis.ConnectionPool(
            host=host, port=port, db=db, password=password,
            decode_responses=decode_responses, socket_keepalive=True,
            health_check_interval=30, max_connections=max_connections,
        )
        self.client = redis.Redis(connection_pool=self._pool)

    # -- low-level ----------------------------------------------------
    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except RedisError:
            return False

    # -- key/value (JSON) --------------------------------------------
    def set_json(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        _retry(lambda: self.client.set(key, json.dumps(value, default=str)))
        if ttl:
            _retry(lambda: self.client.expire(key, ttl))

    def get_json(self, key: str) -> Optional[Any]:
        v = _retry(lambda: self.client.get(key))
        return json.loads(v) if v else None

    # -- streams ------------------------------------------------------
    def xadd(self, stream: str, fields: Dict[str, Any], maxlen: int = 10000) -> str:
        serialised = {k: json.dumps(v, default=str) for k, v in fields.items()}
        return _retry(
            lambda: self.client.xadd(stream, serialised, maxlen=maxlen, approximate=True)
        )

    def xread(
        self,
        streams: Dict[str, str],
        count: int = 100,
        block_ms: Optional[int] = None,
    ) -> List[Tuple[str, List[Tuple[str, Dict[str, str]]]]]:
        return _retry(lambda: self.client.xread(streams, count=count, block=block_ms or None))

    def xread_group(
        self, group: str, consumer: str, streams: Dict[str, str],
        count: int = 100, block_ms: Optional[int] = None,
    ):
        return _retry(
            lambda: self.client.xreadgroup(
                group, consumer, streams, count=count, block=block_ms or None
            )
        )

    def xgroup_create(self, stream: str, group: str, id: str = "$", mkstream: bool = True) -> None:
        try:
            _retry(lambda: self.client.xgroup_create(stream, group, id=id, mkstream=mkstream))
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    def xack(self, stream: str, group: str, *ids: str) -> int:
        return _retry(lambda: self.client.xack(stream, group, *ids))

    # -- pub/sub ------------------------------------------------------
    def publish(self, channel: str, message: Any) -> int:
        return _retry(lambda: self.client.publish(channel, json.dumps(message, default=str)))

    def get_subscriber(self) -> "RedisSubscriber":
        """Return a pub/sub subscriber for this pool."""
        return RedisSubscriber(self._pool)


class RedisSubscriber:
    """Pub/sub subscriber using the shared connection pool."""

    def __init__(self, pool: redis.ConnectionPool):
        self._client = redis.Redis(connection_pool=pool)
        self._pubsub = self._client.pubsub()

    def subscribe(self, channel: str) -> None:
        self._pubsub.subscribe(channel)

    def listen(self) -> Any:
        for msg in self._pubsub.listen():
            if msg["type"] == "message":
                yield json.loads(msg["data"])

    def close(self) -> None:
        self._pubsub.close()



