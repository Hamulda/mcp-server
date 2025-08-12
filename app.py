"""
Flask Backend - Optimized with validation, async processing, and error handling
Implementuje Blueprint architecture, proper logging, CORS, a input validation
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Dict, Any, Optional

from flask import Flask, Blueprint, request, jsonify, current_app
from flask_cors import CORS
from marshmallow import Schema, fields, ValidationError
from werkzeug.exceptions import BadRequest, InternalServerError

# Import našich modulů
try:
    from unified_config import get_config
    from academic_scraper import create_scraping_orchestrator
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unified config not available: {e}")
    UNIFIED_CONFIG_AVAILABLE = False
    # Fallback configuration
    class FallbackConfig:
        sources = {'wikipedia': True, 'pubmed': True, 'openalex': True}
    get_config = lambda: FallbackConfig()

# Nastavení loggingu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validační schéma
class ScrapeRequestSchema(Schema):
    query = fields.Str(required=True, validate=lambda x: len(x.strip()) > 0)
    sources = fields.List(fields.Str(), missing=['wikipedia', 'openalex'])
    max_results = fields.Int(missing=10, validate=lambda x: 1 <= x <= 100)
    timeout = fields.Int(missing=30, validate=lambda x: 5 <= x <= 120)

# Global thread pool pro async processing
executor = ThreadPoolExecutor(max_workers=4)

def async_endpoint(f):
    """Decorator pro asynchronní zpracování v endpointech"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()
    return wrapper

def validate_json(schema_class):
    """Decorator pro validaci JSON inputu"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                schema = schema_class()
                validated_data = schema.load(request.get_json() or {})
                return f(validated_data, *args, **kwargs)
            except ValidationError as e:
                logger.warning(f"Validation error: {e.messages}")
                return jsonify({
                    'error': 'Validation failed',
                    'details': e.messages
                }), 400
            except Exception as e:
                logger.error(f"Unexpected validation error: {e}")
                return jsonify({'error': 'Invalid request format'}), 400
        return wrapper
    return decorator

# Blueprint definice
api_bp = Blueprint('api', __name__, url_prefix='/api')

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })

@api_bp.route('/scrape', methods=['POST'])
@validate_json(ScrapeRequestSchema)
@async_endpoint
async def scrape_endpoint(validated_data: Dict[str, Any]):
    """
    Hlavní scraping endpoint s asynchronním zpracováním
    """
    start_time = time.time()
    query = validated_data['query']
    sources = validated_data['sources']
    max_results = validated_data['max_results']
    timeout = validated_data['timeout']

    logger.info(f"Processing scrape request: query='{query}', sources={sources}")

    try:
        # Inicializace scraperu - použij správnou třídu z academic_scraper.py
        orchestrator = create_scraping_orchestrator()

        # Asynchronní scraping
        results = await orchestrator.scrape_all_sources(
            query=query,
            sources=sources
        )

        processing_time = time.time() - start_time

        # Sestavení odpovědi
        response = {
            'success': True,
            'query': query,
            'sources_requested': sources,
            'processing_time': round(processing_time, 2),
            'results': [
                {
                    'source': r.source,
                    'success': r.success,
                    'data': r.data,
                    'error': r.error,
                    'response_time': r.response_time
                }
                for r in results
            ],
            'summary': {
                'total_sources': len(results),
                'successful_sources': len([r for r in results if r.success]),
                'total_papers': sum(len(r.data.get('papers', [])) for r in results if r.success)
            }
        }

        # Cleanup
        await orchestrator.cleanup()

        logger.info(f"Scrape completed in {processing_time:.2f}s: {len(results)} sources processed")
        return jsonify(response)

    except asyncio.TimeoutError:
        logger.warning(f"Scrape request timed out after {timeout}s")
        return jsonify({
            'error': 'Request timed out',
            'timeout': timeout
        }), 408

    except Exception as e:
        logger.error(f"Scrape request failed: {e}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'message': str(e) if current_app.debug else 'Processing failed'
        }), 500

@api_bp.route('/sources', methods=['GET'])
def get_available_sources():
    """Vrátí seznam dostupných zdrojů"""
    try:
        if UNIFIED_CONFIG_AVAILABLE:
            config = get_config()
            sources = list(config.sources.keys())
        else:
            from config import BaseConfig
            sources = list(BaseConfig.SOURCES.keys())

        return jsonify({
            'sources': sources,
            'count': len(sources)
        })
    except Exception as e:
        logger.error(f"Failed to get sources: {e}")
        return jsonify({'error': 'Failed to retrieve sources'}), 500

@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@api_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@api_bp.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

def create_app(config_name='development'):
    """Application factory"""
    app = Flask(__name__)

    # Konfigurace - použij náš nový config systém
    try:
        # Pokus se použít unified config pouze pokud je dostupný
        if UNIFIED_CONFIG_AVAILABLE:
            config = get_config()
            # Mapování unified_config na náš systém
            app.config['DEBUG'] = getattr(config, 'debug', False)
            app.config['TESTING'] = getattr(config, 'testing', False)
        else:
            # Použij náš nový config systém
            if config_name == 'development':
                from config import DevelopmentConfig
                config_obj = DevelopmentConfig()
                app.config['DEBUG'] = config_obj.APP.debug
                app.config['TESTING'] = config_obj.APP.testing
            elif config_name == 'testing':
                from config import TestingConfig
                config_obj = TestingConfig()
                app.config['DEBUG'] = config_obj.APP.debug
                app.config['TESTING'] = config_obj.APP.testing
            else:
                from config import ProductionConfig
                config_obj = ProductionConfig()
                app.config['DEBUG'] = config_obj.APP.debug
                app.config['TESTING'] = config_obj.APP.testing
    except Exception as e:
        # Fallback na defaults
        logger.warning(f"Config loading failed, using defaults: {e}")
        app.config['DEBUG'] = config_name == 'development'
        app.config['TESTING'] = config_name == 'testing'

    # CORS setup
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:8501"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })

    # Registrace blueprintů
    app.register_blueprint(api_bp)

    # Logging setup
    if not app.testing:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)

    return app

# Pro development server
if __name__ == '__main__':
    app = create_app()
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
