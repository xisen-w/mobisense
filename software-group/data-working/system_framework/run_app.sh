#!/bin/bash

# Run the MobiSense Experiment System Streamlit app
echo "Starting MobiSense Experiment System..."

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found."
    echo "You may need to run setup.sh first or manually install dependencies."
    echo "Continuing without virtual environment..."
fi

# Create experiments directory if it doesn't exist
mkdir -p experiments

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Error: Streamlit is not installed."
    echo "Please run the setup script first: ./setup.sh"
    exit 1
fi

# Run the Streamlit app
echo "Launching Streamlit application..."
streamlit run streamlit_app.py

# Check if the app was able to start
if [ $? -ne 0 ]; then
    echo "Error: Failed to start Streamlit application."
    echo "Please ensure all dependencies are installed correctly."
    echo "Try running the setup script: ./setup.sh"
    exit 1
fi 