#!/bin/bash

# Start MCP Server Script - Spuštění MCP serveru s kontrolou

echo "🚀 Spouštím MCP Server..."

# Kontrola Docker dostupnosti
if ! command -v docker &> /dev/null; then
    echo "❌ Docker není nainstalován nebo dostupný"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose není nainstalován nebo dostupný"
    exit 1
fi

# Zastavení případných běžících kontejnerů
echo "🛑 Zastavujem předchozí instance..."
docker-compose down --remove-orphans

# Kontrola portu 8001
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8001 je obsazený. Pokusím se uvolnit..."
    # Zjistím PID procesu na portu 8001
    PID=$(lsof -ti:8001)
    if [ ! -z "$PID" ]; then
        echo "🔄 Ukončuji proces $PID na portu 8001..."
        kill -9 $PID 2>/dev/null || true
        sleep 2
    fi
fi

# Sestavení a spuštění
echo "🔨 Sestavuji Docker kontejnery..."
docker-compose build

echo "🚀 Spouštím služby..."
docker-compose up -d

# Čekání na spuštění
echo "⏳ Čekám na spuštění serveru..."
sleep 10

# Kontrola health check
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -f http://localhost:8001/health >/dev/null 2>&1; then
        echo "✅ MCP Server je úspěšně spuštěn!"
        echo ""
        echo "📊 Server je dostupný na:"
        echo "   • API Documentation: http://localhost:8001/docs"
        echo "   • MCP Tools: http://localhost:8001/mcp/tools"
        echo "   • Health Check: http://localhost:8001/health"
        echo "   • Prometheus: http://localhost:9091"
        echo "   • Grafana: http://localhost:3001 (admin/admin)"
        echo ""
        echo "🔧 MCP nástroje dostupné pro AI agenty:"
        echo "   • read_file - Čtení souborů"
        echo "   • write_file - Zápis do souborů"
        echo "   • run_in_terminal - Spouštění příkazů"
        echo "   • research - Akademický výzkum"
        echo ""
        echo "📝 Logy můžete sledovat pomocí: docker-compose logs -f"
        exit 0
    fi

    echo "⏳ Pokus $ATTEMPT/$MAX_ATTEMPTS - server se ještě načítá..."
    sleep 3
    ATTEMPT=$((ATTEMPT + 1))
done

echo "❌ Server se nepodařilo spustit v rozumném čase"
echo "📋 Zobrazuji logy pro diagnostiku:"
docker-compose logs --tail=50

exit 1
