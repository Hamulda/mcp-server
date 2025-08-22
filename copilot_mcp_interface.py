#!/usr/bin/env python3
"""
GitHub Copilot MCP Interface - Mé osobní MCP nástroje pro komunikaci
"""

import sys
sys.path.append('/Users/vojtechhamada/PycharmProjects/PythonProject2')

from smart_mcp_tools import SmartMCPTools
from mcp_assistant_bridge import SimpleMCPBridge
from datetime import datetime

class CopilotMCPInterface:
    """Mé osobní MCP rozhraní pro komunikaci s uživateli"""

    def __init__(self):
        self.tools = SmartMCPTools()
        self.bridge = SimpleMCPBridge()
        self.session_id = f"copilot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Inicializace systému
        self._initialize()

    def _initialize(self):
        """Inicializuje mé MCP nástroje"""
        try:
            # Nastavení persistent kontextu
            self.tools.smart_context_manager(
                self.session_id,
                "store",
                {
                    "copilot_status": "active",
                    "session_start": datetime.now().isoformat(),
                    "capabilities": [
                        "intelligent_search",
                        "context_persistence",
                        "academic_analysis",
                        "biohacking_research",
                        "token_optimization"
                    ]
                }
            )
            print(f"✅ GitHub Copilot MCP Interface inicializován - Session: {self.session_id}")
        except Exception as e:
            print(f"⚠️ Inicializace: {e}")

    def search_for_user(self, query: str) -> dict:
        """Provede vyhledávání pro uživatele s optimalizací tokenů"""
        return self.tools.smart_search(query)

    def analyze_academic_content(self, url_or_text: str) -> dict:
        """Analyzuje akademický obsah pro uživatele"""
        return self.tools.academic_paper_analyzer(url_or_text)

    def store_conversation_context(self, context_data: dict):
        """Uloží kontext konverzace pro budoucí použití"""
        return self.tools.smart_context_manager(
            self.session_id,
            "store",
            context_data
        )

    def get_biohacking_resources(self) -> dict:
        """Získá biohacking research nástroje"""
        return self.tools.get_biohacking_research_tools()

# Globální instance pro mé použití
_copilot_mcp = CopilotMCPInterface()

def get_my_mcp_tools():
    """Vrací mé MCP nástroje"""
    return _copilot_mcp

if __name__ == "__main__":
    interface = CopilotMCPInterface()
    print("🤖 GitHub Copilot MCP Interface je aktivní!")
    print(f"📋 Session ID: {interface.session_id}")
    print("🔧 Dostupné nástroje:")
    print("   • Inteligentní vyhledávání")
    print("   • Persistent kontext")
    print("   • Academic analýza")
    print("   • Biohacking research")
    print("   • Token optimalizace")
