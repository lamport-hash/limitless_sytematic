#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

COMPOSE_FILES=(
    "docker-compose-screener.yaml"
    "docker-compose-downloader.yaml"
    "docker-compose-main-app.yaml"
)

build_all() {
    log_info "Building all services..."
    for compose_file in "${COMPOSE_FILES[@]}"; do
        if [[ -f "$compose_file" ]]; then
            log_info "Building from $compose_file..."
            docker compose -f "$compose_file" build --no-cache
        else
            log_warn "$compose_file not found, skipping"
        fi
    done
    log_info "Build complete!"
}

up_all() {
    log_info "Starting all services..."
    for compose_file in "${COMPOSE_FILES[@]}"; do
        if [[ -f "$compose_file" ]]; then
            log_info "Starting from $compose_file..."
            docker compose -f "$compose_file" up -d
        else
            log_warn "$compose_file not found, skipping"
        fi
    done
    log_info "All services started!"
    show_status
}

down_all() {
    log_info "Stopping all services..."
    for compose_file in "${COMPOSE_FILES[@]}"; do
        if [[ -f "$compose_file" ]]; then
            log_info "Stopping from $compose_file..."
            docker compose -f "$compose_file" down
        fi
    done
    log_info "All services stopped!"
}

show_status() {
    log_info "Service status:"
    echo ""
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(candle|listener)" || echo "No services running"
    echo ""
    log_info "Endpoints:"
    echo "  Screener:    http://localhost:8004/screener/data_inventory"
    echo "  Downloader:  http://localhost:7899/download_status"
    echo "  Manager API: http://localhost:7999/health"
}

rebuild_all() {
    down_all
    build_all
    up_all
}

logs() {
    local service=$1
    case "$service" in
        screener)   docker logs -f candle_screener ;;
        downloader) docker logs -f candle_downloader ;;
        manager)    docker logs -f listener-manager-api ;;
        *)          log_error "Unknown service: $service (use: screener, downloader, manager)" ;;
    esac
}

usage() {
    echo "Usage: $0 {build|up|down|restart|status|rebuild|logs <service>}"
    echo ""
    echo "Commands:"
    echo "  build   - Build all Docker images"
    echo "  up      - Start all services"
    echo "  down    - Stop all services"
    echo "  restart - Restart all services"
    echo "  status  - Show service status"
    echo "  rebuild - Stop, rebuild, and start all services"
    echo "  logs    - Follow logs for a service (screener|downloader|manager)"
}

case "${1:-}" in
    build)    build_all ;;
    up)       up_all ;;
    down)     down_all ;;
    restart)  down_all && up_all ;;
    status)   show_status ;;
    rebuild)  rebuild_all ;;
    logs)     logs "$2" ;;
    *)        usage ;;
esac
