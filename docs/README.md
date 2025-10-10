# Financial Knowledge Graph - Complete Setup Guide

An AI-powered pipeline for extracting entities and relationships from financial documents and building interactive knowledge graphs using Neo4j and Streamlit.

## 🚀 Quick Start (Linux)

### Prerequisites Installation

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip curl docker.io docker-compose

# Arch Linux
sudo pacman -S python python-pip curl docker docker-compose

# Fedora/RHEL
sudo dnf install python3 python3-pip curl docker docker-compose

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart terminal
```

### Project Setup

```bash
# Clone/navigate to project directory
cd /path/to/financial-knowledge-graph

# Install all dependencies with uv
uv sync

# Copy environment configuration
cp .env.example .env
# Edit .env with your API keys (see Configuration section)
```

### Neo4j Database Setup (Choose one method)

#### Option 1: Docker (Recommended)
```bash
# Start Neo4j container
docker run -d \
  --name neo4j-financial \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  -v neo4j_data:/data \
  -v neo4j_logs:/logs \
  neo4j:5.15

# Verify it's running
docker ps | grep neo4j
```

#### Option 2: Docker Compose (for development)
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15
    container_name: neo4j-financial
    ports:
      - "7474:7474"  # Web interface
      - "7687:7687"  # Bolt protocol
    environment:
      - NEO4J_AUTH=neo4j/password123
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
```

Then run:
```bash
docker-compose up -d neo4j
```

#### Option 3: Native Installation (Ubuntu/Debian)
```bash
# Add Neo4j repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable latest' | sudo tee /etc/apt/sources.list.d/neo4j.list

# Install Neo4j
sudo apt update
sudo apt install neo4j

# Configure and start
sudo systemctl enable neo4j
sudo systemctl start neo4j

# Set initial password
sudo neo4j-admin set-initial-password password123
```

### Launch the Application

```bash
# Start the Streamlit web interface
uv run streamlit run streamlit_app.py

# Application will be available at:
# http://localhost:8501
```

### Verify Setup

1. **Neo4j Browser**: Visit `http://localhost:7474`
   - Username: `neo4j`
   - Password: `password123`

2. **Streamlit App**: Visit `http://localhost:8501`
   - Should show the Financial Knowledge Graph interface

## 🔧 Configuration

### Environment Variables (.env)

```env
# Neo4j Database Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# LLM Provider Configuration (choose one)
# OpenRouter (recommended for variety)
OPENROUTER_API_KEY=your_openrouter_key_here
LLM_MODEL=google/gemma-2-9b-it:free

# Hugging Face (for open-source models)
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Optional: Advanced settings
CHUNK_SIZE=1000
OVERLAP_SIZE=200
MAX_ENTITIES_PER_CHUNK=50
```

### Free LLM Models Available

**OpenRouter Free Models:**
- `google/gemma-2-9b-it:free`
- `mistralai/mistral-7b-instruct:free`
- `nousresearch/hermes-3-llama-3.1-405b:free`

**Hugging Face Models:**
- Various open-source models available

## 📊 Usage Guide

### 1. Document Processing Tab
- Upload PDF financial documents (earnings reports, SEC filings, etc.)
- Configure extraction parameters
- Process documents to extract entities and relationships

### 2. Graph Building Tab
- Build knowledge graph from processed entities
- View graph statistics and node/relationship counts
- Configure graph building parameters

### 3. Visualization Tab
- Generate interactive network visualizations
- Choose between PyVis (interactive) or Plotly (static)
- Export visualizations as HTML files

### 4. Query Tab
- Ask natural language questions about your data
- Get AI-powered answers based on graph relationships
- Export query results

## 🐧 Linux-Specific Setup Tips

### Docker Setup on Linux

```bash
# Add user to docker group (logout/login required)
sudo usermod -aG docker $USER

# Start Docker service
sudo systemctl enable docker
sudo systemctl start docker

# For Docker Compose v2
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Firewall Configuration

```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 8501  # Streamlit
sudo ufw allow 7474  # Neo4j web
sudo ufw allow 7687  # Neo4j bolt

# Fedora/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --permanent --add-port=7474/tcp
sudo firewall-cmd --permanent --add-port=7687/tcp
sudo firewall-cmd --reload
```

### Resource Management

```bash
# Monitor resource usage
htop
docker stats  # If using Docker

# Increase memory limits for Docker
# Edit /etc/docker/daemon.json
{
  "default-runtime": "runc",
  "default-shm-size": "1G",
  "default-ulimits": {
    "memlock": {
      "Hard": -1,
      "Name": "memlock",
      "Soft": -1
    }
  }
}
```

## 🔧 Development Setup

### Development Tools

```bash
# Install development dependencies
uv sync --dev

# Code formatting and linting
uv run black .
uv run isort .
uv run flake8 .

# Type checking
uv run mypy src/

# Run tests
uv run pytest
uv run pytest --cov=src  # with coverage
```

### Project Structure

```
financial-knowledge-graph/
├── src/                    # Core application modules
│   ├── document_processor/ # PDF processing and text extraction
│   ├── graph_builder/      # Neo4j integration and graph construction
│   ├── llm_clients/        # LLM provider integrations
│   ├── entity_extractor/   # AI-powered entity extraction
│   └── visualization/      # Graph visualization tools
├── data/                   # Sample documents and test data
├── docs/                   # Documentation
├── outputs/               # Generated files and visualizations
├── notebooks/             # Jupyter notebooks for analysis
├── streamlit_app.py       # Main web interface
├── pyproject.toml         # Project configuration and dependencies
├── .env                   # Environment variables
└── docker-compose.yml     # Multi-service Docker setup
```

## 🚨 Troubleshooting

### Common Linux Issues

#### 1. Permission Denied (Docker)
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again

# Or run with sudo (not recommended for production)
sudo docker run ...
```

#### 2. Port Already in Use
```bash
# Check what's using the port
sudo netstat -tlnp | grep :7474
sudo netstat -tlnp | grep :8501

# Kill processes using the ports
sudo kill -9 <PID>

# Or use different ports
docker run -p 7475:7474 -p 7688:7687 ...
```

#### 3. Neo4j Connection Issues
```bash
# Check if Neo4j is running
docker ps | grep neo4j
# or
sudo systemctl status neo4j

# Check logs
docker logs neo4j-financial
# or
sudo journalctl -u neo4j

# Test connection
curl http://localhost:7474
```

#### 4. Python/uv Issues
```bash
# Update uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clear cache and reinstall
uv cache clean
uv sync --reinstall

# Check Python version
python3 --version  # Should be >= 3.8
```

#### 5. Memory/Performance Issues
```bash
# Monitor system resources
free -h
df -h
htop

# For Docker containers
docker stats

# Increase swap if needed (Ubuntu)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Application-Specific Issues

#### LLM API Errors
- Verify API keys in `.env` file
- Check rate limits and quotas
- Use free models for testing:
  - OpenRouter: `google/gemma-2-9b-it:free`
  - Mock provider for offline testing

#### Streamlit Issues
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/

# Run in debug mode
uv run streamlit run streamlit_app.py --logger.level=debug

# Check for port conflicts
netstat -tlnp | grep :8501
```

#### Neo4j Performance
```bash
# Increase heap size (edit neo4j.conf)
dbms.memory.heap.initial_size=512m
dbms.memory.heap.max_size=2G

# For Docker
docker run -e NEO4J_dbms_memory_heap_max__size=2G ...
```

## 📈 Performance Optimization

### For Large Documents
- Process in smaller chunks (adjust `CHUNK_SIZE` in .env)
- Use batch processing for multiple documents
- Monitor memory usage during processing

### For Large Graphs
- Use visualization limits (e.g., top 100 nodes)
- Implement graph sampling for very large datasets
- Consider graph database indexing for query performance

## 🔒 Security Notes

- Keep API keys secure and never commit them to version control
- Use strong passwords for Neo4j in production
- Consider using Docker secrets for sensitive data
- Limit network access in production deployments

## 📚 Additional Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Docker Documentation](https://docs.docker.com/)

---

**Ready to build knowledge graphs from your financial documents!** 🚀📊