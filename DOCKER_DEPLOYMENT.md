# Docker Deployment Guide for TI-Tool

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t ti-tool:latest .
```

### 2. Run with Docker Compose (Recommended)

```bash
# Copy your .env file to the project root (or create one)
cp .env.example .env  # Edit with your credentials

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### 3. Run with Docker Command (Alternative)

```bash
docker run -d \
  --name ti-tool \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/crawled_data:/app/crawled_data \
  -v $(pwd)/processed_data:/app/processed_data \
  -v $(pwd)/summarised_content:/app/summarised_content \
  -v $(pwd)/embeddings_storage:/app/embeddings_storage \
  ti-tool:latest
```

## VM Deployment Instructions

### Prerequisites on VM

1. **Install Docker**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (optional, to run without sudo)
sudo usermod -aG docker $USER
# Log out and back in for this to take effect
```

2. **Verify Installation**:
```bash
docker --version
docker-compose --version
```

### Deployment Steps

1. **Clone or Upload Your Code**:
```bash
# Option A: Clone from GitHub
git clone https://github.com/sharifahshaista/TI-tool.git
cd TI-tool

# Option B: Upload via SCP
scp -r /path/to/TI-tool user@your-vm-ip:/home/user/
```

2. **Create Environment File**:
```bash
# Create .env file with your credentials
nano .env
```

Paste your environment variables:
```env
# LLM Provider
LLM_PROVIDER=azure
EMBEDDING_PROVIDER=azure

# Azure OpenAI
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_MODEL_NAME=pmo-gpt-4.1-nano
AZURE_OPENAI_EMBEDDING_API_KEY=your_key_here
AZURE_OPENAI_EMBEDDING_ENDPOINT=your_endpoint_here
AZURE_OPENAI_EMBEDDING_API_VERSION=2023-05-15
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large

# AWS S3
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=ap-southeast-2
AWS_S3_BUCKET=ti-tool-s3-storage
USE_S3_STORAGE=true

# LinkedIn
LINKEDIN_USERNAME=your_email@example.com
LINKEDIN_PASSWORD=your_password

# SearXNG (optional)
SEARXNG_URL=http://localhost:8888/
```

3. **Build and Run**:
```bash
# Build the image
docker-compose build

# Start in detached mode
docker-compose up -d

# Check if running
docker-compose ps

# View logs
docker-compose logs -f ti-tool
```

4. **Access the Application**:
- Open your browser: `http://your-vm-ip:8501`
- If running locally: `http://localhost:8501`

### Firewall Configuration

If you can't access the app externally, open port 8501:

```bash
# Ubuntu/Debian with UFW
sudo ufw allow 8501/tcp
sudo ufw reload

# Or with iptables
sudo iptables -A INPUT -p tcp --dport 8501 -j ACCEPT
sudo iptables-save
```

## Useful Docker Commands

### Container Management
```bash
# View running containers
docker-compose ps
docker ps

# View all containers (including stopped)
docker ps -a

# Stop the application
docker-compose down

# Restart the application
docker-compose restart

# Stop and remove volumes (careful - deletes data!)
docker-compose down -v
```

### Logs and Debugging
```bash
# View logs
docker-compose logs -f ti-tool

# View last 100 lines
docker-compose logs --tail=100 ti-tool

# Execute commands in running container
docker-compose exec ti-tool /bin/bash

# View container resource usage
docker stats ti-tool
```

### Updating the Application
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Production Best Practices

### 1. Use a Reverse Proxy (Nginx)

Create `nginx.conf`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

### 2. Add SSL with Certbot (HTTPS)
```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### 3. Set Up Auto-restart on System Boot
```bash
# Docker Compose already has restart: unless-stopped
# But also ensure Docker starts on boot:
sudo systemctl enable docker
```

### 4. Monitor with Health Checks
```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' ti-tool

# Set up automated health monitoring
watch -n 30 'curl -f http://localhost:8501/_stcore/health || echo "UNHEALTHY"'
```

### 5. Backup Data Volumes
```bash
# Backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf backup_${DATE}.tar.gz data/ crawled_data/ processed_data/ summarised_content/ embeddings_storage/

# Upload to S3
aws s3 cp backup_${DATE}.tar.gz s3://your-backup-bucket/
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker-compose logs ti-tool

# Check if port is already in use
sudo lsof -i :8501

# Remove and rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Out of Memory
```bash
# Check memory usage
docker stats

# Add resource limits in docker-compose.yml (uncomment deploy section)
# Or increase VM memory
```

### Permission Issues
```bash
# Fix ownership of mounted volumes
sudo chown -R $USER:$USER data/ crawled_data/ processed_data/ summarised_content/
```

### Can't Access from External IP
```bash
# Check if container is listening on all interfaces
docker-compose exec ti-tool netstat -tlnp | grep 8501

# Should show 0.0.0.0:8501, not 127.0.0.1:8501

# Check firewall
sudo ufw status
sudo iptables -L -n | grep 8501
```

## Environment-Specific Notes

### AWS EC2 Deployment
- Use EC2 Instance Role for AWS credentials (more secure than env vars)
- Configure Security Group to allow inbound traffic on port 8501
- Consider using Application Load Balancer for production

### Azure VM Deployment
- Configure Network Security Group (NSG) to allow port 8501
- Use Azure Managed Identity for secure credential management
- Consider Azure Container Instances for easier management

### Google Cloud VM Deployment
- Configure firewall rules to allow tcp:8501
- Use Workload Identity for secure access to GCP services
- Consider Cloud Run for serverless deployment

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review Dockerfile and docker-compose.yml configurations
- Ensure all environment variables are correctly set in `.env`
