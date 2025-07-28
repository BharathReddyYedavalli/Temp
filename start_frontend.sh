#!/bin/bash
echo "Starting Glaucoma Detection Frontend..."
cd "$(dirname "$0")/glaucoma-frontend"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    npm install
fi

# Start the frontend
npm run dev
