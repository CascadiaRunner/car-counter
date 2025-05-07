# Car Counter

A real-time vehicle detection and tracking system using YOLOv8 and OpenCV. This project counts unique vehicles passing through a video feed, with a focus on cars, motorcycles, buses, and trucks.

## Features

- Real-time vehicle detection using YOLOv8
- Multi-object tracking with OpenCV's CSRT tracker
- Unique vehicle counting with ID tracking
- Edge detection to prevent double-counting
- Visual display of tracking boxes and vehicle counts

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- NumPy
- SciPy

## Setup

1. Clone the repository:
```bash
git clone https://github.com/CascadiaRunner/car-counter.git
cd car-counter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the YOLOv8 model (if not already present):
```bash
# The model will be downloaded automatically on first run
```

## Usage

Run the car detector with a video file:
```bash
python car_detector.py
```

Press 'q' to quit the video display.

## How It Works

The system uses YOLOv8 for initial vehicle detection and OpenCV's CSRT tracker for maintaining object tracking between frames. It implements a sophisticated tracking system that:

- Detects vehicles using YOLOv8
- Tracks detected vehicles using CSRT tracker
- Assigns unique IDs to each vehicle
- Prevents double-counting at frame edges
- Maintains a count of unique vehicles

## License

MIT License
