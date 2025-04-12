#!/bin/bash

echo "Setting up MobiSense Experiment System..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r ../../requirements.txt

# Create experiments directory if it doesn't exist
echo "Creating experiments directory..."
mkdir -p experiments

echo "Setup complete! You can now run the application with:"
echo "./run_app.sh" 