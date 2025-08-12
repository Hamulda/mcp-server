.PHONY: help up down restart logs ps rebuild test lint fmt shell metrics

# Default variables (override like: make up COMPOSE=docker-compose)
COMPOSE ?= docker compose
SERVICE ?= research-tool

help:
	@echo "Available targets:"
	@echo "  up        - Build and start all services (detached)"
	@echo "  down      - Stop and remove services and networks"
	@echo "  restart   - Restart the application service"
	@echo "  logs      - Tail logs of the application service"
	@echo "  ps        - Show running containers"
	@echo "  rebuild   - Rebuild images and restart"
	@echo "  test      - Run pytest inside the app container"
	@echo "  lint      - Run flake8 inside the app container"
	@echo "  fmt       - Run black formatter inside the app container"
	@echo "  shell     - Open a shell in the app container"
	@echo "  metrics   - Quick check for /metrics endpoint"

up:
	$(COMPOSE) up -d --build

 down:
	$(COMPOSE) down -v

restart:
	$(COMPOSE) restart $(SERVICE)

logs:
	$(COMPOSE) logs -f $(SERVICE)

ps:
	$(COMPOSE) ps

rebuild:
	$(COMPOSE) build --no-cache
	$(COMPOSE) up -d

# Test, lint and fmt assume the container is running
 test:
	$(COMPOSE) exec $(SERVICE) pytest -q || (echo "Container not running; starting..." && $(COMPOSE) up -d && $(COMPOSE) exec $(SERVICE) pytest -q)

 lint:
	$(COMPOSE) exec $(SERVICE) flake8 || (echo "Container not running; starting..." && $(COMPOSE) up -d && $(COMPOSE) exec $(SERVICE) flake8)

 fmt:
	$(COMPOSE) exec $(SERVICE) black . || (echo "Container not running; starting..." && $(COMPOSE) up -d && $(COMPOSE) exec $(SERVICE) black .)

shell:
	$(COMPOSE) exec $(SERVICE) /bin/sh || $(COMPOSE) exec $(SERVICE) /bin/bash

metrics:
	@echo "GET http://localhost:8000/metrics"
	@curl -sf http://localhost:8000/metrics | head -n 5 || (echo "\nMetrics not available" && exit 1)
