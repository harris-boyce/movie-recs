#!/bin/bash

# Install npm dependencies
cd /workspace/frontend
npm install

# Build for development
npm run build:dev

echo "Frontend environment initialized!"
echo "- Dev server: npm run dev (http://localhost:3000)"