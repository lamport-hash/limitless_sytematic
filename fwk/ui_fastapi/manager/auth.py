from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


class AuthConfig:
    def __init__(self, api_keys: dict[str, dict]):
        self.api_keys = api_keys

    def validate_key(self, api_key: Optional[str]) -> Optional[dict]:
        if not api_key:
            return None
        return self.api_keys.get(api_key)

    def has_role(self, api_key: Optional[str], role: str) -> bool:
        key_info = self.validate_key(api_key)
        if not key_info:
            return False
        return key_info.get("role") == role or key_info.get("role") == "admin"


def create_auth_dependency(auth_config: AuthConfig, allowed_roles: list[str]):
    async def dependency(request: Request, api_key: str = Depends(API_KEY_HEADER)):
        if not auth_config.validate_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

        key_info = auth_config.validate_key(api_key)
        role = key_info.get("role", "")

        if "admin" in allowed_roles and role == "admin":
            return api_key
        if "worker" in allowed_roles and role == "worker":
            return api_key
        if "client" in allowed_roles and role == "client":
            return api_key

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )

    return dependency
