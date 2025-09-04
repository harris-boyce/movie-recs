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
