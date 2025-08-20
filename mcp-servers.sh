#!/bin/bash

# MCP Servers Setup and Management Script
# This script helps manage all installed MCP servers

echo "🔧 MCP Servers Management"
echo "=========================="

# Function to start a specific MCP server
start_server() {
    case $1 in
        "brave-search")
            echo "🔍 Starting Brave Search MCP Server..."
            cd mcp_servers/brave-search-mcp-server
            node dist/index.js
            ;;
        "puppeteer")
            echo "🕷️ Starting Puppeteer MCP Server..."
            cd mcp_servers/servers-archived/src/puppeteer
            node dist/index.js
            ;;
        "git")
            echo "📦 Starting Git MCP Server..."
            cd mcp_servers/mcp-official-servers/src/git
            node dist/index.js
            ;;
        "fetch")
            echo "🌐 Starting Fetch MCP Server..."
            cd mcp_servers/mcp-official-servers/src/fetch
            node dist/index.js
            ;;
        "github")
            echo "🐙 Starting GitHub MCP Server..."
            echo "⚠️  Note: Requires Go to be installed and binary to be built first"
            cd mcp_servers/github-mcp-server
            ./github-mcp-server
            ;;
        *)
            echo "❌ Unknown server: $1"
            echo "Available servers: brave-search, puppeteer, git, fetch, github"
            ;;
    esac
}

# Function to build all servers
build_all() {
    echo "🔨 Building all MCP servers..."

    echo "Building Brave Search MCP Server..."
    cd mcp_servers/brave-search-mcp-server && npm run build
    cd ../..

    echo "Building Puppeteer MCP Server..."
    cd mcp_servers/servers-archived/src/puppeteer && npm run build
    cd ../../../..

    echo "Building Git MCP Server..."
    cd mcp_servers/mcp-official-servers/src/git && npm run build
    cd ../../../..

    echo "Building Fetch MCP Server..."
    cd mcp_servers/mcp-official-servers/src/fetch && npm run build
    cd ../../../..

    echo "✅ All TypeScript servers built successfully!"
    echo "⚠️  GitHub MCP Server requires Go - please install Go and run 'go build' in mcp_servers/github-mcp-server"
}

# Function to list all available servers
list_servers() {
    echo "📋 Available MCP Servers:"
    echo "========================"
    echo "1. 🔍 brave-search    - Brave Search API integration"
    echo "2. 🕷️ puppeteer       - Web scraping and automation"
    echo "3. 📦 git             - Git repository operations"
    echo "4. 🌐 fetch           - HTTP requests and API calls"
    echo "5. 🐙 github          - GitHub API integration (requires Go)"
    echo ""
    echo "Usage: ./mcp-servers.sh [start|build|list] [server-name]"
}

# Main script logic
case $1 in
    "start")
        if [ -z "$2" ]; then
            echo "❌ Please specify a server to start"
            list_servers
        else
            start_server $2
        fi
        ;;
    "build")
        build_all
        ;;
    "list")
        list_servers
        ;;
    *)
        echo "🔧 MCP Servers Management Script"
        echo "Usage: ./mcp-servers.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  start [server-name]  - Start a specific MCP server"
        echo "  build               - Build all MCP servers"
        echo "  list                - List all available servers"
        echo ""
        list_servers
        ;;
esac
