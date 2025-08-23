"""
Enhanced Security Manager - Enterprise-grade bezpečnost
Rate limiting, input validation, audit logging, token management
"""

import secrets
import time
import logging
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from collections import defaultdict
import jwt
from passlib.context import CryptContext
import re

logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Bezpečnostní událost pro audit log"""
    event_type: str
    user_id: Optional[str]
    ip_address: str
    endpoint: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    severity: str = "info"  # info, warning, critical

@dataclass
class RateLimitRule:
    """Pravidlo pro rate limiting"""
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int

class EnhancedSecurityManager:
    """
    Pokročilý security manager s rate limiting a audit logem
    """

    def __init__(self):
        # Password hashing
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # JWT konfigurace
        self.secret_key = self._get_or_generate_secret()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30

        # Rate limiting storage
        self.rate_limit_storage: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        self.blocked_until: Dict[str, datetime] = {}

        # Security events audit log
        self.security_events: List[SecurityEvent] = []
        self.max_events = 10000  # Maximální počet events v paměti

        # Rate limit rules pro různé endpointy
        self.rate_limit_rules = {
            "research": RateLimitRule(20, 100, 5),      # Research queries
            "peptide": RateLimitRule(15, 75, 3),        # Peptide lookups
            "health": RateLimitRule(60, 300, 10),       # Health checks
            "auth": RateLimitRule(5, 20, 2),            # Authentication
            "default": RateLimitRule(30, 150, 5)        # Default rule
        }

        # Bezpečnostní patterns
        self.suspicious_patterns = [
            r'[<>"\']',  # HTML/XSS characters
            r'(union|select|insert|update|delete|drop)\s',  # SQL injection
            r'(javascript:|data:|vbscript:)',  # Script injection
            r'(\.\./){2,}',  # Path traversal
            r'(exec|eval|system|cmd)',  # Command injection
        ]

    def _get_or_generate_secret(self) -> str:
        """Získá nebo vygeneruje tajný klíč"""
        import os

        secret = os.getenv("SECRET_KEY")
        if not secret:
            # Generuj nový secret pro development
            secret = secrets.token_urlsafe(32)
            logger.warning("⚠️ Using generated SECRET_KEY - set SECRET_KEY environment variable for production")

        return secret

    def hash_password(self, password: str) -> str:
        """Hashuje heslo bezpečně"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Ověří heslo proti hashi"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Vytvoří JWT access token"""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})

        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Ověří JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.PyJWTError:
            return None

    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        ip_address: str
    ) -> Dict[str, Any]:
        """
        Kontrola rate limiting

        Args:
            identifier: User ID nebo IP adresa
            endpoint: Název endpointu
            ip_address: IP adresa pro blocking

        Returns:
            Dict s informacemi o rate limitu
        """
        now = time.time()

        # Kontrola blokovaných IP
        if ip_address in self.blocked_ips:
            blocked_until = self.blocked_until.get(ip_address)
            if blocked_until and datetime.now() < blocked_until:
                await self._log_security_event(
                    "rate_limit_blocked",
                    None,
                    ip_address,
                    endpoint,
                    {"reason": "IP temporarily blocked"},
                    "warning"
                )
                return {
                    "allowed": False,
                    "reason": "IP temporarily blocked",
                    "retry_after": int((blocked_until - datetime.now()).total_seconds())
                }
            else:
                # Odblokuj po vypršení
                self.blocked_ips.discard(ip_address)
                self.blocked_until.pop(ip_address, None)

        # Získej rate limit pravidla
        rule = self.rate_limit_rules.get(endpoint, self.rate_limit_rules["default"])

        # Vyčisti staré záznamy
        minute_ago = now - 60
        hour_ago = now - 3600

        requests = self.rate_limit_storage[identifier]
        self.rate_limit_storage[identifier] = [
            req_time for req_time in requests if req_time > hour_ago
        ]

        recent_requests = self.rate_limit_storage[identifier]
        requests_last_minute = sum(1 for req_time in recent_requests if req_time > minute_ago)
        requests_last_hour = len(recent_requests)

        # Kontrola limitů
        if requests_last_minute >= rule.requests_per_minute:
            await self._log_security_event(
                "rate_limit_exceeded",
                identifier,
                ip_address,
                endpoint,
                {
                    "requests_last_minute": requests_last_minute,
                    "limit": rule.requests_per_minute
                },
                "warning"
            )

            # Dočasné blokování při opakovaném překročení
            if requests_last_minute >= rule.requests_per_minute * 2:
                self.blocked_ips.add(ip_address)
                self.blocked_until[ip_address] = datetime.now() + timedelta(minutes=15)

            return {
                "allowed": False,
                "reason": "Rate limit exceeded - too many requests per minute",
                "requests_last_minute": requests_last_minute,
                "limit": rule.requests_per_minute,
                "retry_after": 60
            }

        if requests_last_hour >= rule.requests_per_hour:
            await self._log_security_event(
                "rate_limit_exceeded",
                identifier,
                ip_address,
                endpoint,
                {
                    "requests_last_hour": requests_last_hour,
                    "limit": rule.requests_per_hour
                },
                "warning"
            )

            return {
                "allowed": False,
                "reason": "Rate limit exceeded - too many requests per hour",
                "requests_last_hour": requests_last_hour,
                "limit": rule.requests_per_hour,
                "retry_after": 3600
            }

        # Zaznamenaj request
        self.rate_limit_storage[identifier].append(now)

        return {
            "allowed": True,
            "requests_last_minute": requests_last_minute + 1,
            "requests_last_hour": requests_last_hour + 1,
            "remaining_minute": rule.requests_per_minute - requests_last_minute - 1,
            "remaining_hour": rule.requests_per_hour - requests_last_hour - 1
        }

    def validate_input(self, input_string: str, max_length: int = 1000) -> Dict[str, Any]:
        """
        Komplexní validace vstupů

        Args:
            input_string: Vstupní řetězec
            max_length: Maximální délka

        Returns:
            Dict s výsledkem validace
        """
        # Kontrola délky
        if len(input_string) > max_length:
            return {
                "valid": False,
                "reason": f"Input too long (max {max_length} characters)",
                "sanitized": input_string[:max_length]
            }

        # Kontrola suspektních patterns
        suspicious_found = []
        for pattern in self.suspicious_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                suspicious_found.append(pattern)

        if suspicious_found:
            return {
                "valid": False,
                "reason": "Suspicious patterns detected",
                "patterns": suspicious_found,
                "sanitized": self._sanitize_input(input_string)
            }

        # Sanitizace i pro validní inputy
        sanitized = self._sanitize_input(input_string)

        return {
            "valid": True,
            "sanitized": sanitized,
            "original_length": len(input_string),
            "sanitized_length": len(sanitized)
        }

    def _sanitize_input(self, input_string: str) -> str:
        """Základní sanitizace vstupů"""
        # Odstraň nebezpečné znaky
        sanitized = re.sub(r'[<>"\']', '', input_string)

        # Odstraň SQL injection patterns
        sanitized = re.sub(r'\b(union|select|insert|update|delete|drop)\b', '', sanitized, flags=re.IGNORECASE)

        # Odstraň script injection
        sanitized = re.sub(r'(javascript:|data:|vbscript:)', '', sanitized, flags=re.IGNORECASE)

        # Odstraň path traversal
        sanitized = re.sub(r'\.\./', '', sanitized)

        # Odstraň command injection
        sanitized = re.sub(r'\b(exec|eval|system|cmd)\b', '', sanitized, flags=re.IGNORECASE)

        return sanitized.strip()

    async def _log_security_event(
        self,
        event_type: str,
        user_id: Optional[str],
        ip_address: str,
        endpoint: str,
        details: Dict[str, Any],
        severity: str = "info"
    ):
        """Zaloguje bezpečnostní událost"""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            endpoint=endpoint,
            details=details,
            severity=severity
        )

        self.security_events.append(event)

        # Udržuj maximální počet events
        if len(self.security_events) > self.max_events:
            self.security_events = self.security_events[-self.max_events:]

        # Log podle severity
        log_message = f"Security event: {event_type} from {ip_address} on {endpoint}"
        if severity == "critical":
            logger.critical(log_message, extra={"security_event": event})
        elif severity == "warning":
            logger.warning(log_message, extra={"security_event": event})
        else:
            logger.info(log_message, extra={"security_event": event})

    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generuje bezpečnostní report za posledních N hodin"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]

        # Statistiky podle typu
        event_counts = defaultdict(int)
        ip_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for event in recent_events:
            event_counts[event.event_type] += 1
            ip_counts[event.ip_address] += 1
            severity_counts[event.severity] += 1

        # Top IP adresy s nejvíce events
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Aktuálně blokované IP
        currently_blocked = []
        for ip, blocked_until in self.blocked_until.items():
            if datetime.now() < blocked_until:
                currently_blocked.append({
                    "ip": ip,
                    "blocked_until": blocked_until.isoformat(),
                    "remaining_seconds": int((blocked_until - datetime.now()).total_seconds())
                })

        return {
            "report_period_hours": hours,
            "total_events": len(recent_events),
            "events_by_type": dict(event_counts),
            "events_by_severity": dict(severity_counts),
            "top_ips": [{"ip": ip, "event_count": count} for ip, count in top_ips],
            "currently_blocked_ips": currently_blocked,
            "rate_limit_stats": {
                "total_identifiers_tracked": len(self.rate_limit_storage),
                "blocked_ips_count": len(self.blocked_ips)
            }
        }

    def get_rate_limit_status(self, identifier: str) -> Dict[str, Any]:
        """Vrací aktuální rate limit status pro identifikátor"""
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600

        requests = self.rate_limit_storage.get(identifier, [])

        # Vyčisti staré záznamy
        recent_requests = [req_time for req_time in requests if req_time > hour_ago]
        requests_last_minute = sum(1 for req_time in recent_requests if req_time > minute_ago)
        requests_last_hour = len(recent_requests)

        return {
            "identifier": identifier,
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour,
            "is_blocked": identifier in self.blocked_ips,
            "blocked_until": self.blocked_until.get(identifier, {}).isoformat() if identifier in self.blocked_until else None
        }

# Global security manager instance
_security_manager = None

def get_security_manager() -> EnhancedSecurityManager:
    """Singleton pro security manager"""
    global _security_manager
    if _security_manager is None:
        _security_manager = EnhancedSecurityManager()
    return _security_manager
