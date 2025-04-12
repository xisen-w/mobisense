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
# First check for local requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Using local requirements.txt file..."
    pip install -r requirements.txt
# Then try project root requirements.txt
elif [ -f "../../requirements.txt" ]; then
    echo "Using project requirements.txt file..."
    pip install -r ../../requirements.txt
else
    # Fallback to direct dependency installation
    echo "Installing dependencies directly..."
    pip install streamlit>=1.20.0 pandas>=1.3.0 numpy>=1.20.0 matplotlib>=3.5.0 scipy>=1.7.0 \
    python-dateutil>=2.8.2 pytz>=2021.1 tzdata>=2022.1 pillow>=8.0.0 plotly>=5.3.0 \
    scikit-learn>=1.0.0 watchdog>=2.1.0
fi

# Create experiments directory if it doesn't exist
echo "Creating experiments directory..."
mkdir -p experiments

echo "Setup complete! You can now run the application with:"
echo "./run_app.sh" 