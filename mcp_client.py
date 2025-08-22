#!/usr/bin/env python3
"""
MCP Client - Jednoduchý klient pro komunikaci s MCP servery
"""

import json
import subprocess

def test_mcp_server(server_path, server_args=None):
    """Jednoduchý synchronní test MCP serveru"""
    server_args = server_args or []

    # Příkaz pro spuštění serveru
    cmd = ["node", server_path] + server_args

    # Initialize požadavek
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "PythonProject2-MCP-Client",
                "version": "1.0.0"
            }
        }
    }

    try:
        print(f"🚀 Spouštím MCP server: {' '.join(cmd)}")

        # Spustí proces a pošle initialize request
        process = subprocess.run(
            cmd,
            input=json.dumps(init_request),
            capture_output=True,
            text=True,
            timeout=10
        )

        if process.returncode == 0 and process.stdout:
            print("✅ MCP Server úspěšně odpověděl!")

            # Parsuj odpověď
            try:
                response = json.loads(process.stdout.strip())
                if "result" in response:
                    server_info = response["result"]["serverInfo"]
                    capabilities = response["result"]["capabilities"]

                    print(f"📋 Server: {server_info['name']} v{server_info['version']}")
                    print(f"🔧 Schopnosti: {list(capabilities.keys())}")

                    if "instructions" in response["result"]:
                        print(f"📖 Instrukce: {response['result']['instructions'][:200]}...")

                    return True
                else:
                    print(f"❌ Chybná odpověď: {response}")
                    return False

            except json.JSONDecodeError as e:
                print(f"❌ Chyba při parsování JSON: {e}")
                print(f"Raw output: {process.stdout}")
                return False
        else:
            print(f"❌ Server selhal (kód {process.returncode})")
            if process.stderr:
                print(f"Chyby: {process.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Timeout - server neodpověděl včas")
        return False
    except Exception as e:
        print(f"❌ Neočekávaná chyba: {e}")
        return False

def main():
    """Testuje dostupné MCP servery"""
    print("🔧 MCP Servers Test")
    print("==================")

    # Seznam serverů k testování
    servers = [
        {
            "name": "Everything Server",
            "path": "/Users/vojtechhamada/PycharmProjects/PythonProject2/mcp_servers/mcp-official-servers/src/everything/dist/index.js",
            "args": ["stdio"]
        },
        {
            "name": "Memory Server",
            "path": "/Users/vojtechhamada/PycharmProjects/PythonProject2/mcp_servers/mcp-official-servers/src/memory/dist/index.js",
            "args": []
        },
        {
            "name": "Filesystem Server",
            "path": "/Users/vojtechhamada/PycharmProjects/PythonProject2/mcp_servers/mcp-official-servers/src/filesystem/dist/index.js",
            "args": []
        }
    ]

    working_servers = []

    for server in servers:
        print(f"\n🧪 Testuji {server['name']}...")
        if test_mcp_server(server["path"], server["args"]):
            working_servers.append(server["name"])
        print("-" * 50)

    print(f"\n📊 Výsledky:")
    print(f"✅ Funkční servery: {len(working_servers)}")
    for server_name in working_servers:
        print(f"  - {server_name}")

    if working_servers:
        print(f"\n🎉 Máte {len(working_servers)} funkčních MCP serverů!")
    else:
        print(f"\n😞 Žádné MCP servery nefungují správně.")

if __name__ == "__main__":
    main()
