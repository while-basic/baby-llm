# 🚀 Deployment Guide - Neural Child Development System

**Complete deployment guide for the Neural Child Development System**

---

## 📋 Prerequisites

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- **Git** (for cloning the repository)
- **NVIDIA GPU** (optional, for GPU acceleration)

---

## 🐳 Docker Deployment (Recommended)

### Quick Start

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

**Windows (PowerShell):**
```powershell
.\deploy.ps1
```

### Manual Deployment

1. **Build the Docker image:**
```bash
docker-compose build
```

2. **Start Ollama service:**
```bash
docker-compose up -d ollama
```

3. **Pull the Ollama model:**
```bash
docker exec neural-child-ollama ollama pull gemma3:1b
```

4. **Start all services:**
```bash
docker-compose up -d
```

5. **Check service status:**
```bash
docker-compose ps
```

6. **View logs:**
```bash
docker-compose logs -f
```

---

## 🌐 Accessing the Services

After deployment, the services will be available at:

- **Neural Child Web Interface**: http://localhost:5000
- **Ollama API**: http://localhost:11434
- **Health Check**: http://localhost:5000/api/health

---

## ⚙️ Configuration

### Environment Variables

Set these environment variables before deployment:

```bash
# Flask secret key (required for production)
export FLASK_SECRET_KEY="your-secret-key-here"

# Ollama configuration (optional)
export OLLAMA_HOST="http://ollama:11434"
```

### Configuration File

Edit `config/config.yaml` to customize:
- Ollama model settings
- Neural network parameters
- Development speed
- Memory settings

---

## 🔧 Deployment Modes

### Web Interface Mode

```bash
docker-compose up -d
# Or manually:
docker run -p 5000:5000 neural-child-unified --web
```

### Autonomous Mode

```bash
docker run neural-child-unified --auto
```

### Autonomous Mode with Web Interface

```bash
docker run -p 5000:5000 neural-child-unified --auto-web
```

---

## 📊 Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f neural-child

# Last 100 lines
docker-compose logs --tail=100 neural-child
```

### Health Checks

```bash
# Check service health
curl http://localhost:5000/api/health

# Check Ollama
curl http://localhost:11434/api/tags
```

### Resource Usage

```bash
# Container stats
docker stats

# Disk usage
docker system df
```

---

## 🔄 Updating

### Update the Application

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Update Ollama Model

```bash
docker exec neural-child-ollama ollama pull gemma3:1b
docker-compose restart neural-child
```

---

## 🛑 Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ deletes data)
docker-compose down -v
```

---

## 🐛 Troubleshooting

### Service Won't Start

1. **Check Docker is running:**
```bash
docker ps
```

2. **Check logs:**
```bash
docker-compose logs neural-child
```

3. **Check port availability:**
```bash
# Linux/Mac
lsof -i :5000

# Windows
netstat -ano | findstr :5000
```

### Ollama Connection Issues

1. **Verify Ollama is running:**
```bash
docker-compose ps ollama
```

2. **Check Ollama logs:**
```bash
docker-compose logs ollama
```

3. **Test Ollama connection:**
```bash
curl http://localhost:11434/api/tags
```

### GPU Issues

If you have an NVIDIA GPU, uncomment the GPU configuration in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 📦 Production Deployment

### Security Considerations

1. **Set a strong Flask secret key:**
```bash
export FLASK_SECRET_KEY=$(openssl rand -hex 32)
```

2. **Use environment variables for sensitive data:**
```bash
# Create .env file
cat > .env << EOF
FLASK_SECRET_KEY=your-secret-key
OLLAMA_HOST=http://ollama:11434
EOF
```

3. **Use a reverse proxy (nginx/traefik) for HTTPS**

4. **Limit container resources:**
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

### Scaling

To run multiple instances:

```bash
docker-compose up -d --scale neural-child=3
```

Use a load balancer (nginx/traefik) to distribute traffic.

---

## ☁️ Cloud Deployment

### AWS (EC2/ECS)

1. **Build and push to ECR:**
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag neural-child-unified:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/neural-child:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/neural-child:latest
```

2. **Deploy using ECS with Fargate or EC2**

### Google Cloud Platform

1. **Build and push to GCR:**
```bash
gcloud builds submit --tag gcr.io/<project-id>/neural-child
gcloud run deploy neural-child --image gcr.io/<project-id>/neural-child
```

### Azure

1. **Build and push to ACR:**
```bash
az acr build --registry <registry-name> --image neural-child:latest .
az container create --resource-group <rg-name> --name neural-child --image <registry-name>.azurecr.io/neural-child:latest
```

---

## 📝 Maintenance

### Backup Data

```bash
# Backup volumes
docker run --rm -v neural-child-unified_data:/data -v $(pwd):/backup alpine tar czf /backup/data-backup.tar.gz /data
```

### Clean Up

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a --volumes
```

---

## 🆘 Support

For issues or questions:
- Check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) guide
- Review logs: `docker-compose logs`
- Open an issue on GitHub

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**
