"""
Rate Limiting Middleware for UltraThink Pilot microservices.

Implements sliding window rate limiting with in-memory storage.
For production, consider Redis-backed rate limiting for distributed deployments.

Usage:
    from common_utils.rate_limiter import rate_limit_middleware

    app.middleware("http")(rate_limit_middleware)
"""

from fastapi import Request, HTTPException, status
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    In-memory rate limiter using sliding window algorithm.

    Tracks requests per identifier (API key or IP) and enforces
    configurable limits per minute.
    """

    def __init__(self, requests_per_minute: int = 100):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute (default: 100)
        """
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
        logger.info(f"Rate limiter initialized: {requests_per_minute} requests/min")

    async def check_rate_limit(self, identifier: str):
        """
        Check if request is within rate limit.

        Args:
            identifier: Unique identifier (API key or IP address)

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        async with self.lock:
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean old requests (sliding window)
            self.requests[identifier] = [
                ts for ts in self.requests[identifier]
                if ts > minute_ago
            ]

            # Check limit
            if len(self.requests[identifier]) >= self.requests_per_minute:
                logger.warning(
                    f"Rate limit exceeded for {identifier[:10]}...: "
                    f"{len(self.requests[identifier])} requests in last minute"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                    headers={"Retry-After": "60"}
                )

            # Record request
            self.requests[identifier].append(now)


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=100)


async def rate_limit_middleware(request: Request, call_next):
    """
    FastAPI middleware for rate limiting.

    Applies rate limiting based on API key (if present) or IP address.
    Health check endpoints are exempt from rate limiting.

    Args:
        request: FastAPI request object
        call_next: Next middleware in chain

    Returns:
        Response from next middleware/endpoint

    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    # Exempt health checks from rate limiting
    if request.url.path.endswith('/health'):
        response = await call_next(request)
        return response

    # Get identifier (prefer API key, fallback to IP)
    identifier = request.headers.get('X-API-Key')
    if not identifier:
        identifier = request.client.host if request.client else 'unknown'

    # Check rate limit
    await rate_limiter.check_rate_limit(identifier)

    # Process request
    response = await call_next(request)
    return response
