#!/bin/bash

# Restore .NET packages
cd /workspace/dotnet-model-api
dotnet restore

# Build solution
dotnet build

# Wait for models to be available
echo "Waiting for ML models..."
while [ ! -f "/models/movie_taste.onnx" ]; do
    sleep 5
done

# Validate model
dotnet run --project src/MovieModel.Api -- --validate-model

echo ".NET backend initialized!"
echo "- API will be available at: http://localhost:5000"