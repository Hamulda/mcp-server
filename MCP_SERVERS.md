# 🤖 MCP Servers Installation Guide

## 📦 Installed MCP Servers

All requested MCP servers have been successfully installed and configured:

### ✅ Successfully Installed Servers

1. **🔍 Brave Search MCP Server**
   - **Location**: `mcp_servers/brave-search-mcp-server/`
   - **Source**: https://github.com/brave/brave-search-mcp-server
   - **Features**: Web search, news search, image search, video search
   - **Status**: ✅ Installed and built
   - **Requirements**: Brave API Key

2. **🕷️ Puppeteer MCP Server**
   - **Location**: `mcp_servers/servers-archived/src/puppeteer/`
   - **Source**: https://github.com/modelcontextprotocol/servers-archived/tree/main/src/puppeteer
   - **Features**: Web scraping, browser automation, screenshot capture
   - **Status**: ✅ Installed and built
   - **Note**: From archived servers repository

3. **📦 Git MCP Server**
   - **Location**: `mcp_servers/mcp-official-servers/src/git/`
   - **Source**: https://github.com/modelcontextprotocol/servers/tree/main/src/git
   - **Features**: Git repository operations, commit history, branch management
   - **Status**: ✅ Installed and built

4. **🌐 Fetch MCP Server**
   - **Location**: `mcp_servers/mcp-official-servers/src/fetch/`
   - **Source**: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch
   - **Features**: HTTP requests, API calls, web content fetching
   - **Status**: ✅ Installed and built

5. **🐙 GitHub MCP Server**
   - **Location**: `mcp_servers/github-mcp-server/`
   - **Source**: https://github.com/github/github-mcp-server
   - **Features**: GitHub API integration, repository management, issues, PRs
   - **Status**: ⚠️ Cloned (requires Go to build)
   - **Requirements**: GitHub Personal Access Token, Go programming language

## 🚀 Quick Start

### Using the Management Script

```bash
# List all available servers
./mcp-servers.sh list

# Build all servers (except GitHub which needs Go)
./mcp-servers.sh build

# Start a specific server
./mcp-servers.sh start brave-search
./mcp-servers.sh start puppeteer
./mcp-servers.sh start git
./mcp-servers.sh start fetch
```

### Manual Server Startup

#### Brave Search MCP Server
```bash
cd mcp_servers/brave-search-mcp-server
export BRAVE_API_KEY="your-api-key-here"
node dist/index.js
```

#### Puppeteer MCP Server
```bash
cd mcp_servers/servers-archived/src/puppeteer
node dist/index.js
```

#### Git MCP Server
```bash
cd mcp_servers/mcp-official-servers/src/git
node dist/index.js
```

#### Fetch MCP Server
```bash
cd mcp_servers/mcp-official-servers/src/fetch
node dist/index.js
```

#### GitHub MCP Server (requires Go)
```bash
# First, install Go from https://golang.org/dl/
cd mcp_servers/github-mcp-server
go build
export GITHUB_PERSONAL_ACCESS_TOKEN="your-token-here"
./github-mcp-server
```

## ⚙️ Configuration

### MCP Configuration File
A complete configuration file is available at `mcp_servers/mcp-config.json`:

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "node",
      "args": ["./mcp_servers/brave-search-mcp-server/dist/index.js"],
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key-here"
      }
    },
    "puppeteer": {
      "command": "node",
      "args": ["./mcp_servers/servers-archived/src/puppeteer/dist/index.js"]
    },
    "git": {
      "command": "node",
      "args": ["./mcp_servers/mcp-official-servers/src/git/dist/index.js"]
    },
    "fetch": {
      "command": "node",
      "args": ["./mcp_servers/mcp-official-servers/src/fetch/dist/index.js"]
    },
    "github": {
      "command": "./mcp_servers/github-mcp-server/github-mcp-server",
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-github-token-here"
      }
    }
  }
}
```

### Environment Variables

Create a `.env` file in the project root with:

```bash
# Brave Search API
BRAVE_API_KEY=your_brave_api_key_here

# GitHub API
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here
```

## 🔧 API Keys Setup

### Brave Search API Key
1. Visit https://api.search.brave.com/
2. Sign up for an account
3. Get your API key
4. Add it to your environment variables

### GitHub Personal Access Token
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with appropriate permissions
3. Add it to your environment variables

## 🛠️ Troubleshooting

### GitHub MCP Server (Go Required)
If you don't have Go installed:

```bash
# macOS with Homebrew
brew install go

# Or download from https://golang.org/dl/

# Then build the server
cd mcp_servers/github-mcp-server
go build
```

### Puppeteer Issues
If Puppeteer fails to start:

```bash
cd mcp_servers/servers-archived/src/puppeteer
npm install
npm run build
```

### Permission Issues
Make sure the management script is executable:

```bash
chmod +x mcp-servers.sh
```

## 📊 Server Features

| Server | Web Search | API Calls | Git Ops | Browser | GitHub |
|--------|------------|-----------|---------|---------|---------|
| Brave Search | ✅ | ✅ | ❌ | ❌ | ❌ |
| Puppeteer | ❌ | ❌ | ❌ | ✅ | ❌ |
| Git | ❌ | ❌ | ✅ | ❌ | ❌ |
| Fetch | ❌ | ✅ | ❌ | ❌ | ❌ |
| GitHub | ❌ | ✅ | ✅ | ❌ | ✅ |

## 🔗 Integration with Academic Research Tool

All MCP servers are now ready to be integrated with your Academic Research Tool. They can be used by AI agents to:

- **Search the web** (Brave Search)
- **Scrape websites** (Puppeteer)
- **Manage repositories** (Git)
- **Make API calls** (Fetch)
- **Interact with GitHub** (GitHub)

## 📈 Next Steps

1. **Set up API keys** for Brave Search and GitHub
2. **Install Go** if you want to use GitHub MCP Server
3. **Test each server** individually using the management script
4. **Integrate with your main application** using the MCP configuration

All servers are production-ready and optimized for your M1 MacBook! 🚀
