#!/bin/bash
#
# UltraThink Pilot - Production Infrastructure Deployment Script
# This script installs Docker and deploys all infrastructure services
#

set -e  # Exit on error

echo "=========================================="
echo "UltraThink Pilot Infrastructure Deployment"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[i]${NC} $1"
}

# Check if running in WSL
if ! grep -q Microsoft /proc/version; then
    print_error "This script must be run in WSL2"
    exit 1
fi

print_status "Running in WSL2 environment"

# =============================================================================
# PHASE 1: Docker Installation
# =============================================================================

echo ""
echo "=== Phase 1: Docker Installation ==="
echo ""

# Check if Docker is already installed
if command -v docker &> /dev/null; then
    print_warning "Docker is already installed"
    docker --version
    read -p "Reinstall Docker? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Skipping Docker installation"
        SKIP_DOCKER=true
    else
        SKIP_DOCKER=false
    fi
else
    SKIP_DOCKER=false
fi

if [ "$SKIP_DOCKER" = false ]; then
    print_info "Updating package index..."
    sudo apt-get update

    print_info "Installing prerequisites..."
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    print_info "Adding Docker's official GPG key..."
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    print_info "Setting up Docker repository..."
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    print_info "Installing Docker Engine..."
    sudo apt-get update
    sudo apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin

    print_status "Docker installed successfully"

    # Start Docker service
    print_info "Starting Docker service..."
    sudo service docker start

    # Verify Docker installation
    print_info "Verifying Docker installation..."
    sudo docker run hello-world

    print_status "Docker is working correctly"

    # Add user to docker group
    print_info "Adding user to docker group..."
    sudo groupadd -f docker
    sudo usermod -aG docker $USER

    print_warning "You may need to log out and back in for docker group changes to take effect"
    print_warning "Or run: newgrp docker"
fi

# Verify Docker Compose
print_info "Checking Docker Compose..."
sudo docker compose version
print_status "Docker Compose is available"

# =============================================================================
# PHASE 2: Environment Configuration
# =============================================================================

echo ""
echo "=== Phase 2: Environment Configuration ==="
echo ""

cd ~/ultrathink-pilot/infrastructure

if [ ! -f .env ]; then
    print_info "Creating .env file from template..."
    cp .env.example .env

    print_warning "IMPORTANT: Please edit .env and set secure passwords!"
    print_info "Required changes:"
    echo "  - POSTGRES_PASSWORD (currently: changeme_in_production)"
    echo "  - GRAFANA_PASSWORD (currently: admin)"

    read -p "Edit .env now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ${EDITOR:-nano} .env
    else
        print_warning "Remember to edit .env before production use!"
    fi
else
    print_status ".env file already exists"
fi

# =============================================================================
# PHASE 3: Start Infrastructure Services
# =============================================================================

echo ""
echo "=== Phase 3: Starting Infrastructure Services ==="
echo ""

print_info "Starting Docker Compose services..."
sudo docker compose up -d

print_status "Services starting..."

# Wait for services to be healthy
print_info "Waiting for services to be healthy (30 seconds)..."
sleep 30

print_info "Checking service status..."
sudo docker compose ps

# =============================================================================
# PHASE 4: Verify Services
# =============================================================================

echo ""
echo "=== Phase 4: Service Verification ==="
echo ""

# Check TimescaleDB
print_info "Checking TimescaleDB..."
if sudo docker exec ultrathink-timescaledb pg_isready -U ultrathink > /dev/null 2>&1; then
    print_status "TimescaleDB is ready"
else
    print_error "TimescaleDB is not ready"
fi

# Check Redis
print_info "Checking Redis..."
if sudo docker exec ultrathink-redis redis-cli ping > /dev/null 2>&1; then
    print_status "Redis is ready"
else
    print_error "Redis is not ready"
fi

# Check container health
print_info "Container status:"
sudo docker compose ps

echo ""
print_status "Infrastructure deployment complete!"
echo ""

# =============================================================================
# PHASE 5: Display Access Information
# =============================================================================

echo "=== Service Access Information ==="
echo ""
echo "  MLflow:      http://localhost:5000"
echo "  Grafana:     http://localhost:3000 (admin/admin)"
echo "  Prometheus:  http://localhost:9090"
echo "  TimescaleDB: localhost:5432 (user: ultrathink)"
echo "  Redis:       localhost:6379"
echo ""

# =============================================================================
# PHASE 6: Next Steps
# =============================================================================

echo "=== Next Steps ==="
echo ""
echo "1. Run database migration:"
echo "   cd ~/ultrathink-pilot"
echo "   source venv/bin/activate"
echo "   python scripts/migrate_sqlite_to_timescale.py"
echo ""
echo "2. Run production training test:"
echo "   python train_professional_v2.py --episodes 10"
echo ""
echo "3. Access Grafana to set up dashboards:"
echo "   http://localhost:3000"
echo ""
echo "4. View logs:"
echo "   sudo docker compose logs -f [service_name]"
echo ""
echo "5. Stop services:"
echo "   cd ~/ultrathink-pilot/infrastructure"
echo "   sudo docker compose down"
echo ""

print_status "Deployment script completed!"
