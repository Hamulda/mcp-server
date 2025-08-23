"""
Enterprise Security Manager - JWT autentifikace a bezpečnostní funkce
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import re
import asyncio
from collections import defaultdict

class SecurityManager:
    def __init__(self):
        self.secret_key = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 60
        self.refresh_token_expire_days = 7
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer()

        # Rate limiting
        self.rate_limit_requests = defaultdict(list)
        self.max_requests_per_minute = 60

        # API key management
        self.api_keys = {}
        self.blacklisted_tokens = set()

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            if token in self.blacklisted_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )

            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input to prevent injection attacks"""
        # Remove potential SQL injection patterns
        sql_patterns = [
            r"(\bUNION\b|\bSELECT\b|\bINSERT\b|\bDELETE\b|\bUPDATE\b|\bDROP\b)",
            r"(\-\-|\#|\/\*|\*\/)",
            r"(\bOR\b|\bAND\b)\s+\d+\s*=\s*\d+"
        ]

        for pattern in sql_patterns:
            input_data = re.sub(pattern, "", input_data, flags=re.IGNORECASE)

        # Remove script tags and potential XSS
        xss_patterns = [
            r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
            r"javascript:",
            r"on\w+\s*="
        ]

        for pattern in xss_patterns:
            input_data = re.sub(pattern, "", input_data, flags=re.IGNORECASE)

        return input_data.strip()

    def check_rate_limit(self, client_ip: str) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Clean old requests
        self.rate_limit_requests[client_ip] = [
            req_time for req_time in self.rate_limit_requests[client_ip]
            if req_time > minute_ago
        ]

        # Check current count
        if len(self.rate_limit_requests[client_ip]) >= self.max_requests_per_minute:
            return False

        # Add current request
        self.rate_limit_requests[client_ip].append(now)
        return True

    def generate_api_key(self, user_id: str, permissions: List[str] = None) -> str:
        """Generate API key for user"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions or [],
            "created_at": datetime.utcnow(),
            "last_used": None
        }
        return api_key

    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["last_used"] = datetime.utcnow()
            return self.api_keys[api_key]
        return None

    def revoke_token(self, token: str) -> None:
        """Add token to blacklist"""
        self.blacklisted_tokens.add(token)

    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """FastAPI dependency to get current user from token"""
        token = credentials.credentials
        payload = self.verify_token(token)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return {"user_id": user_id, "payload": payload}

    def validate_input_length(self, input_data: str, max_length: int = 1000) -> str:
        """Validate and truncate input length"""
        if len(input_data) > max_length:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input too long. Maximum {max_length} characters allowed."
            )
        return input_data

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host

# Global security manager instance
security_manager = SecurityManager()
