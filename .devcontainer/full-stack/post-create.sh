#!/bin/bash

echo "🚀 Initializing full-stack MovieRecs environment..."

# Install Python ML dependencies
echo "🐍 Setting up Python ML environment..."
cd /workspace/python-ml
pip install -e .

# Install .NET dependencies
echo "⚡ Setting up .NET environment..."
cd /workspace/dotnet-model-api
dotnet restore
dotnet build

# Install Node.js dependencies
echo "⚛️ Setting up Node.js environment..."
cd /workspace/frontend
npm install

# Initialize data if needed
echo "📊 Checking for sample data..."
if [ ! -f "/data/movies.csv" ]; then
    echo "📥 Sample data will be downloaded on first ML run"
fi

# Set up git hooks for responsible AI
echo "🔒 Setting up git hooks for responsible AI..."
cd /workspace
if [ -d .git ]; then
    # Create pre-commit hook for model validation
    cat > .git/hooks/pre-commit << 'HOOK_EOF'
#!/bin/bash
# Responsible AI pre-commit hook

echo "🔍 Running responsible AI checks..."

# Check for model files in commits
if git diff --cached --name-only | grep -E "\.(onnx|pkl|h5|pt)$"; then
    echo "⚠️  Model files detected in commit"
    echo "   Please ensure models are validated and documented"
    echo "   Run: python python-ml/scripts/validate_model.py"
fi

# Check for bias testing
if git diff --cached --name-only | grep -E "train|model" | grep "\.py$"; then
    echo "🧪 Model training code changed - remember to update bias tests"
fi

echo "✅ Pre-commit checks completed"
HOOK_EOF
    chmod +x .git/hooks/pre-commit
fi

echo ""
echo "✅ Full-stack environment initialized!"
echo ""
echo "🌐 Available services:"
echo "   - Jupyter Lab: http://localhost:8888"
echo "   - MLflow UI: http://localhost:5555"  
echo "   - Weaviate: http://localhost:8080"
echo "   - Vector Service: http://localhost:8081"
echo ""
echo "🚀 Development commands:"
echo "   - Start frontend: cd frontend && npm run dev"
echo "   - Start .NET API: cd dotnet-model-api && dotnet run"
echo "   - Start Jupyter: jupyter lab --ip=0.0.0.0 --allow-root"
echo ""
echo "📚 Next steps:"
echo "   1. Review docs/development/getting-started.md"
echo "   2. Run sample training: cd python-ml && python scripts/train_pipeline.py"
echo "   3. Validate model: python scripts/validate_model.py"
echo "   4. Start frontend development!"