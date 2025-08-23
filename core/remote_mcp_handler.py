"""
Remote MCP Handler - HTTP/SSE transport pro enterprise deployment
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse
import aiohttp
from security.security_manager import security_manager

logger = logging.getLogger(__name__)

class TokenManager:
    """Správa tokenů pro multi-tenant architekturu"""

    def __init__(self):
        self.user_tokens = {}
        self.token_permissions = {}

    def create_user_token(self, user_id: str, permissions: List[str]) -> str:
        """Vytvoří token pro uživatele s oprávněními"""
        token = security_manager.generate_api_key(user_id, permissions)
        self.user_tokens[user_id] = token
        self.token_permissions[token] = permissions
        return token

    def validate_token_permission(self, token: str, required_permission: str) -> bool:
        """Ověří, zda token má požadované oprávnění"""
        permissions = self.token_permissions.get(token, [])
        return required_permission in permissions or "admin" in permissions

class HTTPTransport:
    """HTTP transport pro remote MCP komunikaci"""

    def __init__(self):
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=30)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_request(self, url: str, method: str, data: Dict[str, Any], headers: Dict[str, str] = None) -> Dict[str, Any]:
        """Pošle HTTP request na remote MCP server"""
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                headers=headers or {}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(status_code=response.status, detail=await response.text())
        except Exception as e:
            logger.error(f"HTTP transport error: {e}")
            raise

class RemoteMCPHandler:
    """Pokročilý MCP handler s remote podporou"""

    def __init__(self):
        self.transport = HTTPTransport()
        self.auth_manager = TokenManager()
        self.connected_clients = {}
        self.audit_log = []
        self.tool_registry = {}

    async def handle_remote_connection(self, user_token: str, client_id: str) -> Dict[str, Any]:
        """Zpracuje remote MCP připojení"""
        try:
            # Ověř token
            user_info = security_manager.validate_api_key(user_token)
            if not user_info:
                raise HTTPException(status_code=401, detail="Invalid token")

            # Registruj klienta
            self.connected_clients[client_id] = {
                "user_id": user_info["user_id"],
                "connected_at": datetime.utcnow(),
                "permissions": user_info["permissions"],
                "last_activity": datetime.utcnow()
            }

            logger.info(f"Remote MCP client connected: {client_id} for user {user_info['user_id']}")

            return {
                "status": "connected",
                "client_id": client_id,
                "available_tools": list(self.tool_registry.keys()),
                "permissions": user_info["permissions"]
            }

        except Exception as e:
            logger.error(f"Remote connection error: {e}")
            raise

    async def execute_tool(self, client_id: str, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Spustí nástroj pro konkrétního klienta"""
        if client_id not in self.connected_clients:
            raise HTTPException(status_code=404, detail="Client not found")

        client_info = self.connected_clients[client_id]

        # Ověř oprávnění
        if not self.auth_manager.validate_token_permission(client_id, f"use_{tool_name}"):
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        # Sanitizuj vstupy
        sanitized_params = {}
        for key, value in parameters.items():
            if isinstance(value, str):
                sanitized_params[key] = security_manager.sanitize_input(value)
            else:
                sanitized_params[key] = value

        # Auditní log
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_id": client_id,
            "user_id": client_info["user_id"],
            "tool_name": tool_name,
            "parameters": sanitized_params,
            "status": "executing"
        }
        self.audit_log.append(audit_entry)

        try:
            # Spusť nástroj
            if tool_name in self.tool_registry:
                result = await self.tool_registry[tool_name](sanitized_params)
                audit_entry["status"] = "completed"
                audit_entry["result_size"] = len(str(result))

                # Aktualizuj poslední aktivitu
                client_info["last_activity"] = datetime.utcnow()

                return {
                    "tool": tool_name,
                    "result": result,
                    "execution_time": (datetime.utcnow() - datetime.fromisoformat(audit_entry["timestamp"].replace('Z', '+00:00'))).total_seconds(),
                    "status": "success"
                }
            else:
                raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")

        except Exception as e:
            audit_entry["status"] = "failed"
            audit_entry["error"] = str(e)
            logger.error(f"Tool execution error: {e}")
            raise

    def register_tool(self, name: str, handler: Callable) -> None:
        """Registruje nástroj"""
        self.tool_registry[name] = handler
        logger.info(f"Registered MCP tool: {name}")

    async def get_audit_log(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Vrátí auditní log pro uživatele"""
        user_logs = [
            log for log in self.audit_log
            if log["user_id"] == user_id
        ]
        return user_logs[-limit:]

    async def stream_events(self, client_id: str):
        """Stream events přes SSE"""
        if client_id not in self.connected_clients:
            raise HTTPException(status_code=404, detail="Client not found")

        async def event_generator():
            while client_id in self.connected_clients:
                # Pošli heartbeat
                yield {
                    "event": "heartbeat",
                    "data": json.dumps({
                        "timestamp": datetime.utcnow().isoformat(),
                        "client_id": client_id
                    })
                }
                await asyncio.sleep(30)  # Heartbeat každých 30 sekund

        return EventSourceResponse(event_generator())

    async def disconnect_client(self, client_id: str) -> None:
        """Odpojí klienta"""
        if client_id in self.connected_clients:
            user_id = self.connected_clients[client_id]["user_id"]
            del self.connected_clients[client_id]
            logger.info(f"Disconnected MCP client: {client_id} for user {user_id}")

# Globální instance
remote_mcp_handler = RemoteMCPHandler()

# FastAPI endpoints pro remote MCP
def setup_remote_mcp_routes(app: FastAPI):
    """Nastavení remote MCP routes"""

    @app.post("/mcp/connect")
    async def connect_remote_mcp(
        request: Request,
        user_token: str,
        current_user = Depends(security_manager.get_current_user)
    ):
        """Připojení remote MCP klienta"""
        client_ip = security_manager.get_client_ip(request)

        # Rate limiting
        if not security_manager.check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        client_id = f"{current_user['user_id']}_{datetime.utcnow().timestamp()}"
        return await remote_mcp_handler.handle_remote_connection(user_token, client_id)

    @app.post("/mcp/execute/{client_id}")
    async def execute_mcp_tool(
        client_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        current_user = Depends(security_manager.get_current_user)
    ):
        """Spuštění MCP nástroje"""
        return await remote_mcp_handler.execute_tool(client_id, tool_name, parameters)

    @app.get("/mcp/stream/{client_id}")
    async def stream_mcp_events(client_id: str):
        """SSE stream pro MCP události"""
        return await remote_mcp_handler.stream_events(client_id)

    @app.delete("/mcp/disconnect/{client_id}")
    async def disconnect_mcp_client(
        client_id: str,
        current_user = Depends(security_manager.get_current_user)
    ):
        """Odpojení MCP klienta"""
        await remote_mcp_handler.disconnect_client(client_id)
        return {"status": "disconnected", "client_id": client_id}

    @app.get("/mcp/audit")
    async def get_mcp_audit_log(
        limit: int = 100,
        current_user = Depends(security_manager.get_current_user)
    ):
        """Auditní log MCP operací"""
        return await remote_mcp_handler.get_audit_log(current_user["user_id"], limit)
