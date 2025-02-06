import redis
import json
import logging
from typing import Any, Optional
from ..config.settings import REDIS_CONFIG, CACHE_EXPIRY

class CacheManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.redis_client = redis.Redis(
                host=REDIS_CONFIG['host'],
                port=REDIS_CONFIG['port'],
                db=REDIS_CONFIG['db'],
                password=REDIS_CONFIG['password'],
                decode_responses=True
            )
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.error(f"Redis connection failed: {e}")
            self.redis_client = None

    def set(self, key: str, value: Any, expiry: int = CACHE_EXPIRY) -> bool:
        try:
            if self.redis_client:
                serialized_value = json.dumps(value)
                return self.redis_client.setex(key, expiry, serialized_value)
            return False
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False

    def get(self, key: str) -> Optional[Any]:
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                return json.loads(value) if value else None
            return None
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None

    def delete(self, key: str) -> bool:
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            return False
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        try:
            if self.redis_client:
                return bool(self.redis_client.flushdb())
            return False
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return False
