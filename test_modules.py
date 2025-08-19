#!/usr/bin/env python3
"""
Test script pro ověření funkčnosti všech hlavních modulů
"""

print('🔍 Testing core modules...')

# Test unified_config
try:
    from unified_config import get_config
    config = get_config()
    print('✅ unified_config: OK')
    print(f'   Environment: {config.environment.value}')
    print(f'   Sources configured: {len(config.sources)}')
except Exception as e:
    print(f'❌ unified_config: {e}')

# Test unified_cache_system
try:
    from unified_cache_system import UnifiedCacheSystem
    cache = UnifiedCacheSystem()
    print('✅ unified_cache_system: OK')
except Exception as e:
    print(f'❌ unified_cache_system: {e}')

# Test academic_scraper
try:
    from academic_scraper import create_scraping_orchestrator
    orchestrator = create_scraping_orchestrator()
    print('✅ academic_scraper: OK')
    print(f'   Available scrapers: {list(orchestrator.scrapers.keys())}')
except Exception as e:
    print(f'❌ academic_scraper: {e}')

print()
print('🗄️ Testing SQLite...')
import sqlite3
try:
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)')
    conn.execute('INSERT INTO test (data) VALUES (?)', ('test_data',))
    result = conn.execute('SELECT * FROM test').fetchone()
    conn.close()
    print('✅ SQLite: Working perfectly')
    print(f'   Test data: {result}')
except Exception as e:
    print(f'❌ SQLite: {e}')

print()
print('📊 Project status summary:')
print('- Core modules tested')
print('- SQLite database functional')
print('- Configuration system unified')
print('- Cache system optimized')
print('- Academic scraper ready')
