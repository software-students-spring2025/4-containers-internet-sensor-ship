[![Web App CI](https://github.com/software-students-spring2025/4-containers-internet-sensor-ship/actions/workflows/build-frontend.yml/badge.svg)](https://github.com/software-students-spring2025/4-containers-internet-sensor-ship/actions/workflows/build-frontend.yml)
[![Machine Learning Client](https://github.com/software-students-spring2025/4-containers-internet-sensor-ship/actions/workflows/build-backend.yml/badge.svg)](https://github.com/software-students-spring2025/4-containers-internet-sensor-ship/actions/workflows/build-backend.yml)
![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)

# Cat Food Monitor 🐱

A containerized system that monitors your cat's eating habits using machine learning and provides a web interface to track feeding patterns.

## Overview

This system consists of three main components:

1. **Machine Learning Client**: Uses computer vision to detect when your cat is eating from their food bowl
2. **Web Application**: Provides a dashboard to view feeding patterns and statistics
3. **MongoDB Database**: Stores all feeding events and user data

## Team

This project was developed by:

- Nick ([@NMichael111](https://github.com/NMichael111))
- Isaac ([@isaac1000000](https://github.com/isaac1000000))

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/software-students-spring2025/4-containers-internet-sensor-ship.git
   cd 4-containers-internet-sensor-ship
   ```

2. The project includes a `.env` file with the required environment variables:
   ```
   MONGODB_URI=mongodb://mongodb:27017/
   MONGODB_DBNAME=cat_monitor
   SECRET_KEY=your-secret-key-here
   ```

3. Build and start all containers:
   ```bash
   docker-compose up --build
   ```

4. Access the web interface at `http://localhost:5000`

5. Create an account then login

6. Click start camera then click enable detection

7. If a cat is detected by the camera then it will be logged and displayed on the graph.

## Using the Web Interface

After logging in, you'll see the main dashboard with:

1. **Recent Feedings**: Shows a list of recent cat feeding events with timestamps and thumbnail images
2. **Authentication**: The system requires login to access the dashboard
3. **Auto-refresh**: The dashboard automatically refreshes to show new feeding events
