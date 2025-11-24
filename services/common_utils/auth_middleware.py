"""
FastAPI Authentication Middleware
Provides API key validation for UltraThink Pilot microservices.

Usage:
    from common_utils.auth_middleware import verify_api_key

    @app.get("/api/v1/endpoint", dependencies=[Depends(verify_api_key)])
    async def protected_endpoint():
        return {"status": "authenticated"}
"""

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os
import logging

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify API key from request header.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        str: The validated API key

    Raises:
        HTTPException: 401 if key is missing or invalid
        HTTPException: 500 if authentication is not configured
    """
    if api_key is None:
        logger.warning("API key missing from request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )

    # Get expected key from environment
    expected_key = os.environ.get('SERVICE_API_KEY')
    if not expected_key:
        logger.error("SERVICE_API_KEY not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured"
        )

    if api_key != expected_key:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return api_key
