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
