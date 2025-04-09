[![Web App CI](https://github.com/software-students-spring2025/4-containers-internet-sensor-ship/actions/workflows/build-frontend.yml/badge.svg)](https://github.com/software-students-spring2025/4-containers-internet-sensor-ship/actions/workflows/build-frontend.yml)
[![Machine Learning Client](https://github.com/software-students-spring2025/4-containers-internet-sensor-ship/actions/workflows/build-backend.yml/badge.svg)](https://github.com/software-students-spring2025/4-containers-internet-sensor-ship/actions/workflows/build-backend.yml)
![Lint-free](https://github.com/nyu-software-engineering/containerized-app-exercise/actions/workflows/lint.yml/badge.svg)


# Cat Food Monitor 🐱

A containerized system that monitors your cat's eating habits using machine learning and provides a web interface to track feeding patterns.

[![Machine Learning Client CI](https://github.com/yourusername/4-containers-internet-sensor-ship/actions/workflows/ml-client-ci.yml/badge.svg)](https://github.com/yourusername/4-containers-internet-sensor-ship/actions/workflows/ml-client-ci.yml)
[![Web App CI](https://github.com/yourusername/4-containers-internet-sensor-ship/actions/workflows/web-app-ci.yml/badge.svg)](https://github.com/yourusername/4-containers-internet-sensor-ship/actions/workflows/web-app-ci.yml)

## Overview

This system consists of three main components:

1. **Machine Learning Client**: Uses computer vision to detect when your cat is eating from their food bowl
2. **Web Application**: Provides a dashboard to view feeding patterns and statistics
3. **MongoDB Database**: Stores all feeding events and user data

## Features

- Real-time cat detection using computer vision
- Daily and weekly feeding statistics
- Feeding timeline visualization
- User authentication system
- Historical data analysis

## System Requirements

- Docker
- Docker Compose
- Python 3.8+
- A camera connected to your system

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/4-containers-internet-sensor-ship.git
   cd 4-containers-internet-sensor-ship
   ```

2. Create a `.env` file in the root directory with the following content:
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

## Development

### Machine Learning Client

Located in the `machine-learning-client` directory, this component:
- Uses OpenCV for camera access
- Implements a simple CNN for cat detection
- Stores detection events in MongoDB

### Web Application

Located in the `web-app` directory, this component:
- Provides a Flask-based web interface
- Implements user authentication
- Visualizes feeding data
- Allows configuration of monitoring settings

## Database Schema

The MongoDB database contains the following collections:

- `users`: User account information
- `feeding_events`: Timestamps and metadata of detected feeding events
- `settings`: System configuration settings

## Contributing

1. Create a feature branch
2. Make your changes
3. Submit a pull request
4. Ensure all tests pass
5. Get code review approval
6. Merge into main

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
