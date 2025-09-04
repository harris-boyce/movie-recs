#!/bin/bash

# MovieRecs Repository Structure Generator
# Creates the complete directory structure for the MovieRecs GenAI application

set -e  # Exit on any error

PROJECT_NAME="movie-recs"

echo "🎬 Creating MovieRecs repository structure..."
echo "📁 Project: $PROJECT_NAME"

# Create project root
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# GitHub Actions & CI/CD
echo "📋 Creating CI/CD structure..."
mkdir -p .github/workflows

# DevContainer configurations
echo "🐳 Creating DevContainer configurations..."
mkdir -p .devcontainer/ml-python
mkdir -p .devcontainer/backend-dotnet
mkdir -p .devcontainer/frontend-node
mkdir -p .devcontainer/full-stack

# Docker configurations
echo "🐳 Creating Docker configurations..."
mkdir -p docker/mlflow
mkdir -p docker/weaviate
mkdir -p docker/faiss-service
mkdir -p docker/nginx

# Persistent volumes (critical for reproducibility)
echo "💾 Creating persistent volume directories..."
mkdir -p volumes/mlflow
mkdir -p volumes/models
mkdir -p volumes/data
mkdir -p volumes/vector-db

# Data directories
echo "📊 Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/schemas

# Python ML tier
echo "🐍 Creating Python ML structure..."
mkdir -p python-ml/src/data
mkdir -p python-ml/src/models
mkdir -p python-ml/src/features
mkdir -p python-ml/src/utils
mkdir -p python-ml/notebooks
mkdir -p python-ml/tests
mkdir -p python-ml/scripts

# .NET Model API tier
echo "⚡ Creating .NET API structure..."
mkdir -p dotnet-model-api/src/MovieModel.Api/Controllers
mkdir -p dotnet-model-api/src/MovieModel.Api/Services
mkdir -p dotnet-model-api/src/MovieModel.Api/Models
mkdir -p dotnet-model-api/src/MovieModel.Api/Configuration
mkdir -p dotnet-model-api/src/MovieModel.Api/Middleware
mkdir -p dotnet-model-api/tests/MovieModel.Api.Tests
mkdir -p dotnet-model-api/tests/MovieModel.Api.IntegrationTests

# Vector service
echo "🔍 Creating vector service structure..."
mkdir -p vector-service/src
mkdir -p vector-service/tests
mkdir -p vector-service/scripts

# Frontend React tier
echo "⚛️ Creating frontend structure..."
mkdir -p frontend/src/components/chat
mkdir -p frontend/src/components/recommendations
mkdir -p frontend/src/components/common
mkdir -p frontend/src/lib
mkdir -p frontend/src/hooks
mkdir -p frontend/src/styles
mkdir -p frontend/public
mkdir -p frontend/tests

# Examples and documentation
echo "📖 Creating documentation structure..."
mkdir -p examples/api_examples
mkdir -p examples/screenshots
mkdir -p docs/architecture
mkdir -p docs/development
mkdir -p docs/api
mkdir -p docs/responsible-ai
mkdir -p docs/deployment

# Infrastructure
echo "🏗️ Creating infrastructure structure..."
mkdir -p infra/k8s/services
mkdir -p infra/terraform

# Utility scripts
echo "🔧 Creating scripts directory..."
mkdir -p scripts

# Create essential placeholder files
echo "📄 Creating essential files..."

# Root level files
touch README.md
touch LICENSE
touch .gitignore

# Environment files
cat > .env.example << 'EOF'
# MLflow Configuration
MLFLOW_TRACKING_URI=file:./volumes/mlflow
MLFLOW_EXPERIMENT_NAME=movie-recs-dev

# Model Configuration  
MODEL_PATH=./volumes/models
MODEL_VALIDATION_ENABLED=true

# Vector Database
VECTOR_DB_TYPE=weaviate
WEAVIATE_URL=http://localhost:8080
FAISS_INDEX_PATH=./volumes/vector-db/faiss.index

# OpenAI (for embeddings)
OPENAI_API_KEY=your-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# API Configuration
API_BASE_URL=http://localhost:5000
CORS_ORIGINS=http://localhost:3000

# Logging
LOG_LEVEL=INFO
REQUEST_LOGGING_ENABLED=true
EOF

# Create basic .gitignore
cat > .gitignore << 'EOF'
# Environment files
.env
.env.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.coverage
.pytest_cache/
*.egg-info/
dist/
build/

# .NET
bin/
obj/
*.user
*.suo
*.cache
*.dll
*.pdb
*.exe
.vs/
.vscode/settings.json

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.eslintcache

# Data files
data/raw/
*.csv
*.json
*.parquet
!examples/**/*.json

# Model artifacts
*.onnx
*.pkl
*.joblib
*.h5
*.pt
*.pth

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Docker
.dockerignore

# Volumes (persistent data)
volumes/mlflow/*
volumes/models/*
volumes/data/*
volumes/vector-db/*
!volumes/.gitkeep

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE
.idea/
*.swp
*.swo
*~

# Logs
logs/
*.log
EOF

# Create volume .gitkeep files to preserve structure
touch volumes/mlflow/.gitkeep
touch volumes/models/.gitkeep
touch volumes/data/.gitkeep
touch volumes/vector-db/.gitkeep

# Python requirements placeholder
cat > python-ml/requirements.txt << 'EOF'
# Core ML/Data Science
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
scipy>=1.9.0

# Deep Learning & Model Export
torch>=1.13.0
onnx>=1.12.0
onnxruntime>=1.13.0

# MLflow & Experiment Tracking
mlflow>=2.0.0

# Vector Search & Embeddings
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0

# Data Processing
pyarrow>=9.0.0
polars>=0.15.0

# Development
jupyter>=1.0.0
jupyterlab>=3.5.0
pytest>=7.0.0
black>=22.0.0
flake8>=5.0.0
pre-commit>=2.20.0
EOF

# Package.json placeholder for frontend
cat > frontend/package.json << 'EOF'
{
  "name": "movie-recs-frontend",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "lint:fix": "eslint . --ext ts,tsx --fix"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "langchain": "^0.1.0",
    "@langchain/core": "^0.1.0",
    "axios": "^1.6.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "@vitejs/plugin-react": "^4.0.0",
    "eslint": "^8.45.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-react-refresh": "^0.4.0",
    "typescript": "^5.0.0",
    "vite": "^4.4.0"
  }
}
EOF

# Basic pyproject.toml for Python ML
cat > python-ml/pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "movie-recs-ml"
version = "0.1.0"
description = "Machine Learning components for MovieRecs GenAI application"
requires-python = ">=3.9"
dependencies = [
    "pandas>=1.5.0",
    "numpy>=1.21.0",
    "scikit-learn>=1.1.0",
    "mlflow>=2.0.0",
    "onnx>=1.12.0",
    "onnxruntime>=1.13.0"
]

[project.optional-dependencies]
dev = [
    "jupyter",
    "pytest",
    "black",
    "flake8",
    "pre-commit"
]

[tool.setuptools]
packages = ["src"]

[tool.black]
line-length = 88
target-version = ['py39', 'py310', 'py311']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
EOF

# .NET project file placeholder
cat > dotnet-model-api/MovieModel.Api.sln << 'EOF'
Microsoft Visual Studio Solution File, Format Version 12.00
# Visual Studio Version 17
VisualStudioVersion = 17.0.31903.59
MinimumVisualStudioVersion = 10.0.40219.1
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "MovieModel.Api", "src\MovieModel.Api\MovieModel.Api.csproj", "{12345678-1234-5678-9012-123456789012}"
EndProject
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "MovieModel.Api.Tests", "tests\MovieModel.Api.Tests\MovieModel.Api.Tests.csproj", "{12345678-1234-5678-9012-123456789013}"
EndProject
Global
	GlobalSection(SolutionConfigurationPlatforms) = preSolution
		Debug|Any CPU = Debug|Any CPU
		Release|Any CPU = Release|Any CPU
	EndGlobalSection
	GlobalSection(ProjectConfigurationPlatforms) = postSolution
		{12345678-1234-5678-9012-123456789012}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
		{12345678-1234-5678-9012-123456789012}.Debug|Any CPU.Build.0 = Debug|Any CPU
		{12345678-1234-5678-9012-123456789012}.Release|Any CPU.ActiveCfg = Release|Any CPU
		{12345678-1234-5678-9012-123456789012}.Release|Any CPU.Build.0 = Release|Any CPU
		{12345678-1234-5678-9012-123456789013}.Debug|Any CPU.ActiveCfg = Debug|Any CPU
		{12345678-1234-5678-9012-123456789013}.Debug|Any CPU.Build.0 = Debug|Any CPU
		{12345678-1234-5678-9012-123456789013}.Release|Any CPU.ActiveCfg = Release|Any CPU
		{12345678-1234-5678-9012-123456789013}.Release|Any CPU.Build.0 = Release|Any CPU
	EndGlobalSection
EndGlobal
EOF

# Quick setup script
cat > scripts/setup-dev.sh << 'EOF'
#!/bin/bash
# Quick development environment setup

echo "🚀 Setting up MovieRecs development environment..."

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file from template"
fi

# Initialize git if not already
if [ ! -d .git ]; then
    git init
    git add .
    git commit -m "Initial commit: MovieRecs repository structure"
    echo "📦 Initialized git repository"
fi

# Create Python virtual environment
if [ ! -d python-ml/.venv ]; then
    cd python-ml
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    cd ..
    echo "🐍 Created Python virtual environment"
fi

echo "✅ Development environment ready!"
echo "🔗 Next steps:"
echo "   1. Open in VS Code with Dev Containers extension"
echo "   2. Choose your development tier:"
echo "      - ML: .devcontainer/ml-python/devcontainer.json"
echo "      - API: .devcontainer/backend-dotnet/devcontainer.json" 
echo "      - Frontend: .devcontainer/frontend-node/devcontainer.json"
echo "      - Full Stack: .devcontainer/full-stack/devcontainer.json"
EOF

chmod +x scripts/setup-dev.sh

# README with quick start
cat > README.md << 'EOF'
# 🎬 MovieRecs - Responsible GenAI Reference Implementation

A polyglot, enterprise-ready reference implementation demonstrating how to combine ML models, vector search, and LLM orchestration for explainable movie recommendations.

## 🎯 Architecture Overview

```
Frontend (React + LangChain.js) → .NET Model API → Python ML Models
                                ↗ Vector Database ↘ MLflow Tracking
```

## 🚀 Quick Start

### Option 1: Complete Setup
```bash
curl -sSL https://raw.githubusercontent.com/your-org/movie-recs/main/create-repo-structure.sh | bash
cd movie-recs
./scripts/setup-dev.sh
```

### Option 2: Choose Your Development Tier
```bash
# ML/Data Science
gh codespace create --devcontainer-path .devcontainer/ml-python/devcontainer.json

# Backend/.NET API  
gh codespace create --devcontainer-path .devcontainer/backend-dotnet/devcontainer.json

# Frontend/React
gh codespace create --devcontainer-path .devcontainer/frontend-node/devcontainer.json

# Full Stack
gh codespace create --devcontainer-path .devcontainer/full-stack/devcontainer.json
```

## 🔬 Responsible AI Features

- **Model Governance**: MLflow tracking with complete experiment lineage
- **Explainability**: Model attribution and recommendation provenance  
- **Validation Pipeline**: Automated model quality gates
- **Bias Testing**: Built-in fairness evaluation framework
- **Transparency**: Clear model-to-prediction traceability

## 📚 Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Development Guide](docs/development/getting-started.md)
- [Responsible AI Guidelines](docs/responsible-ai/fairness-considerations.md)
- [API Documentation](docs/api/model-api.md)

## 🤝 Contributing

This project emphasizes responsible AI practices. Please review our [Responsible AI Guidelines](docs/responsible-ai/) before contributing.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
EOF

echo ""
echo "✅ Repository structure created successfully!"
echo ""
echo "📁 Created project: $PROJECT_NAME/"
echo "📊 Total directories: $(find . -type d | wc -l)"
echo "📄 Total files: $(find . -type f | wc -l)"
echo ""
echo "🚀 Next steps:"
echo "   1. cd $PROJECT_NAME"
echo "   2. ./scripts/setup-dev.sh"
echo "   3. Open in VS Code with Dev Containers"
echo ""
echo "🎓 Happy learning and building responsible AI! 🤖✨"
