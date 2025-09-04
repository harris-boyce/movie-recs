FROM mcr.microsoft.com/devcontainers/dotnet:8.0

# Install Node.js runtime for frontend serving capability
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs

# Install additional tools
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /models /mlflow /workspace

# Set working directory
WORKDIR /workspace

# Expose ports
EXPOSE 5000 5001 8080