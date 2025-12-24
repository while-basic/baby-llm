# 🐳 Docker Setup Guide

**Quick guide to set up Docker for Neural Child Development System**

---

## ⚠️ Common Issue: Docker Not Running

If you see this error:
```
error during connect: Head "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping": 
open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

**This means Docker Desktop is not running!**

---

## 🪟 Windows Setup

### Step 1: Install Docker Desktop

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Run the installer
3. Follow the installation wizard
4. Restart your computer if prompted

### Step 2: Start Docker Desktop

1. **Open Docker Desktop** from the Start menu
2. **Wait for it to fully start** - Look for:
   - The whale icon in the system tray (bottom right)
   - "Docker Desktop is running" message in the app
   - No error messages

3. **Verify Docker is running:**
   ```bash
   docker info
   ```
   
   If this command works without errors, Docker is running!

### Step 3: Run Deployment

Once Docker Desktop is running:
```bash
./deploy.sh
# Or on PowerShell:
.\deploy.ps1
```

---

## 🐧 Linux Setup

### Install Docker

**Ubuntu/Debian:**
```bash
# Update package index
sudo apt-get update

# Install Docker
sudo apt-get install -y docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (optional, to avoid sudo)
sudo usermod -aG docker $USER
# Log out and back in for this to take effect
```

**Verify installation:**
```bash
docker --version
docker info
```

---

## 🍎 macOS Setup

### Install Docker Desktop

1. Download from: https://www.docker.com/products/docker-desktop/
2. Install the `.dmg` file
3. Open Docker Desktop from Applications
4. Wait for it to start (whale icon in menu bar)

### Verify

```bash
docker info
```

---

## ✅ Quick Verification

Run these commands to verify Docker is working:

```bash
# Check Docker version
docker --version

# Check Docker is running
docker info

# Check Docker Compose
docker-compose --version
# Or
docker compose version
```

All commands should work without errors.

---

## 🔧 Troubleshooting

### Docker Desktop Won't Start (Windows)

1. **Check Windows features:**
   - Open "Turn Windows features on or off"
   - Ensure "Virtual Machine Platform" and "Windows Subsystem for Linux" are enabled
   - Restart if you made changes

2. **Check WSL 2:**
   ```powershell
   wsl --status
   ```
   If not using WSL 2:
   ```powershell
   wsl --set-default-version 2
   ```

3. **Restart Docker Desktop:**
   - Right-click the whale icon
   - Select "Restart Docker Desktop"

### Permission Denied (Linux)

If you get "permission denied" errors:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or:
newgrp docker

# Verify
docker ps
```

### Port Already in Use

If port 5000 or 11434 is already in use:

1. **Find what's using the port:**
   ```bash
   # Windows
   netstat -ano | findstr :5000
   
   # Linux/Mac
   lsof -i :5000
   ```

2. **Change ports in docker-compose.yml:**
   ```yaml
   ports:
     - "5001:5000"  # Use 5001 instead of 5000
   ```

---

## 📚 Next Steps

Once Docker is running:

1. **Run the deployment:**
   ```bash
   ./deploy.sh
   ```

2. **Access the services:**
   - Web Interface: http://localhost:5000
   - Ollama API: http://localhost:11434

3. **Check logs:**
   ```bash
   docker-compose logs -f
   ```

---

## 🆘 Still Having Issues?

1. Check Docker Desktop logs:
   - Windows: Right-click whale icon → Troubleshoot
   - Mac: Docker Desktop → Preferences → Troubleshoot

2. Restart Docker Desktop completely

3. Check system requirements:
   - Windows: Windows 10/11 64-bit, WSL 2 enabled
   - Mac: macOS 10.15+ (Catalina or newer)
   - Linux: Kernel 3.10+ with cgroups and namespaces

4. Review [DEPLOYMENT.md](DEPLOYMENT.md) for more details

---

**Built with 🤍 by [Celaya Solutions](https://celayasolutions.com)**
