#!/usr/bin/env python3
"""
MCP Assistant Bridge - Jednoduchý a spolehlivý přístup k MCP serverům
"""

import json
import subprocess
import os
from typing import Dict, Any, List

class SimpleMCPBridge:
    """Jednoduchý MCP bridge pro GitHub Copilot"""

    def __init__(self):
        self.base_path = "/Users/vojtechhamada/PycharmProjects/PythonProject2/mcp_servers"

    def call_mcp_server(self, server_path: str, method: str = "initialize", params: Dict = None) -> Dict[str, Any]:
        """Jednoduché volání MCP serveru"""
        if params is None:
            params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "GitHub-Copilot", "version": "1.0.0"}
            }

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }

        try:
            cmd = ["node", server_path]
            result = subprocess.run(
                cmd,
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=15,
                cwd=os.path.dirname(server_path)
            )

            if result.returncode == 0 and result.stdout.strip():
                try:
                    response = json.loads(result.stdout.strip())
                    return response
                except json.JSONDecodeError:
                    # Někdy servery vrací více řádků - vezmi poslední JSON
                    lines = result.stdout.strip().split('\n')
                    for line in reversed(lines):
                        if line.strip():
                            try:
                                return json.loads(line.strip())
                            except:
                                continue

            return {"error": f"Server error: {result.stderr}"}

        except Exception as e:
            return {"error": f"Call failed: {str(e)}"}

    def call_mcp_server_with_args(self, server_path: str, args: List[str], method: str = "initialize", params: Dict = None) -> Dict[str, Any]:
        """Volání MCP serveru s konkrétními argumenty"""
        if params is None:
            params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "GitHub-Copilot", "version": "1.0.0"}
            }

        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }

        try:
            cmd = ["node", server_path] + args
            result = subprocess.run(
                cmd,
                input=json.dumps(request),
                capture_output=True,
                text=True,
                timeout=15,
                cwd=os.path.dirname(server_path)
            )

            if result.returncode == 0 and result.stdout.strip():
                try:
                    response = json.loads(result.stdout.strip())
                    return response
                except json.JSONDecodeError:
                    # Někdy servery vrací více řádků - vezmi poslední JSON
                    lines = result.stdout.strip().split('\n')
                    for line in reversed(lines):
                        if line.strip():
                            try:
                                return json.loads(line.strip())
                            except:
                                continue

            return {"error": f"Server error: {result.stderr}"}

        except Exception as e:
            return {"error": f"Call failed: {str(e)}"}

# Globální instance pro můj přístup
_bridge = SimpleMCPBridge()

def test_my_mcp_servers():
    """Test všech dostupných MCP serverů"""
    servers = {
        "Everything Server": {
            "path": f"{_bridge.base_path}/mcp-official-servers/src/everything/dist/index.js",
            "args": ["stdio"]
        },
        "Filesystem Server": {
            "path": f"{_bridge.base_path}/mcp-official-servers/src/filesystem/dist/index.js",
            "args": ["/Users/vojtechhamada/PycharmProjects/PythonProject2"]
        },
        "Memory Server": {
            "path": f"{_bridge.base_path}/mcp-official-servers/src/memory/dist/index.js",
            "args": []
        },
        "Sequential Thinking": {
            "path": f"{_bridge.base_path}/mcp-official-servers/src/sequentialthinking/dist/index.js",
            "args": []
        }
    }

    print("🤖 GitHub Copilot MCP Test")
    print("==========================")

    working_servers = []

    for name, config in servers.items():
        path = config["path"]
        args = config["args"]

        if os.path.exists(path):
            print(f"\n🧪 Testing {name}...")
            response = _bridge.call_mcp_server_with_args(path, args)

            if "result" in response:
                server_info = response["result"].get("serverInfo", {})
                capabilities = response["result"].get("capabilities", {})

                print(f"   ✅ {server_info.get('name', name)}")
                print(f"   📋 Capabilities: {list(capabilities.keys())}")
                working_servers.append(name)

                # Test tools list
                tools_response = _bridge.call_mcp_server_with_args(path, args, "tools/list", {})
                if "result" in tools_response:
                    tools = tools_response["result"].get("tools", [])
                    print(f"   🔧 Tools: {len(tools)}")
                    if tools:
                        print(f"      Examples: {[t['name'] for t in tools[:3]]}")

            else:
                print(f"   ❌ Failed: {response.get('error', 'Unknown error')}")
        else:
            print(f"\n❌ {name}: File not found at {path}")

    print(f"\n🎉 Result: {len(working_servers)}/{len(servers)} MCP servers are working!")
    if working_servers:
        print("✅ I now have access to these MCP servers:")
        for server in working_servers:
            print(f"   - {server}")

    return working_servers

# Funkce pro mé vlastní použití
def use_everything_server(tool_name: str, arguments: Dict = None):
    """Použije Everything server pro konkrétní úkol"""
    path = f"{_bridge.base_path}/mcp-official-servers/src/everything/dist/index.js"
    params = {"name": tool_name, "arguments": arguments or {}}
    return _bridge.call_mcp_server(path, "tools/call", params)

def use_filesystem_server(action: str, arguments: Dict = None):
    """Použije Filesystem server pro práci se soubory"""
    path = f"{_bridge.base_path}/mcp-official-servers/src/filesystem/dist/index.js"
    params = {"name": action, "arguments": arguments or {}}
    return _bridge.call_mcp_server(path, "tools/call", params)

def use_memory_server(action: str, arguments: Dict = None):
    """Použije Memory server pro ukládání informací"""
    path = f"{_bridge.base_path}/mcp-official-servers/src/memory/dist/index.js"
    params = {"name": action, "arguments": arguments or {}}
    return _bridge.call_mcp_server(path, "tools/call", params)

if __name__ == "__main__":
    test_my_mcp_servers()
