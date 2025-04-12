#!/bin/bash

# Run the MobiSense Experiment System Streamlit app
echo "Starting MobiSense Experiment System..."

# Create experiments directory if it doesn't exist
mkdir -p experiments

# Run the Streamlit app
streamlit run streamlit_app.py 