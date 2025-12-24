#!/bin/bash
#----------------------------------------------------------------------------
#File:       deploy.sh
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Deployment script for neural child system
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

set -e

echo "============================================================"
echo "Neural Child Development System - Deployment Script"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker is not running.${NC}"
    echo -e "${YELLOW}Please start Docker Desktop and try again.${NC}"
    echo ""
    echo "On Windows:"
    echo "  1. Open Docker Desktop application"
    echo "  2. Wait for it to fully start (whale icon in system tray)"
    echo "  3. Run this script again"
    echo ""
    echo "On Linux/Mac:"
    echo "  1. Start Docker service: sudo systemctl start docker"
    echo "  2. Or start Docker Desktop application"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p data/checkpoints data/logs data/models data/cache
mkdir -p checkpoints memories development_results
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Set Flask secret key if not set
if [ -z "$FLASK_SECRET_KEY" ]; then
    export FLASK_SECRET_KEY=$(openssl rand -hex 32)
    echo -e "${YELLOW}Generated FLASK_SECRET_KEY (set this in production):${NC}"
    echo "export FLASK_SECRET_KEY=$FLASK_SECRET_KEY"
    echo ""
fi

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker-compose build
echo -e "${GREEN}✓ Docker image built${NC}"
echo ""

# Pull Ollama model
echo -e "${YELLOW}Pulling Ollama model (gemma3:1b)...${NC}"
docker-compose up -d ollama
sleep 5
docker exec neural-child-ollama ollama pull gemma3:1b
echo -e "${GREEN}✓ Ollama model pulled${NC}"
echo ""

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose up -d
echo -e "${GREEN}✓ Services started${NC}"
echo ""

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check health
if curl -f http://localhost:5000/api/health &> /dev/null; then
    echo -e "${GREEN}✓ Neural Child service is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Service may still be starting. Check logs with: docker-compose logs${NC}"
fi

echo ""
echo "============================================================"
echo -e "${GREEN}Deployment Complete!${NC}"
echo "============================================================"
echo ""
echo "Services:"
echo "  - Neural Child Web Interface: http://localhost:5000"
echo "  - Ollama API: http://localhost:11434"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo "  - View status: docker-compose ps"
echo ""
echo "============================================================"
