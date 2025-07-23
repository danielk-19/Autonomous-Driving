# CARLA Autonomous Driving System

A hybrid machine learning + rule-based autonomous driving system for CARLA simulator with comprehensive data collection, training, and evaluation capabilities.

## Features

- **Hybrid Agent**: Combines ML perception with rule-based safety systems
- **Multi-Sensor Data Collection**: RGB, semantic segmentation, LiDAR, GPS
- **Comprehensive Evaluation**: Safety, efficiency, and comfort metrics
- **Real-World Applicability**: Safety-first design with hard rule constraints
- **Data Management**: Complete pipeline from collection to training

## Project Structure

```
carla_autonomous_driving/
├── sensors/
│   └── capture_sensors.py      # Sensor data collection and management
├── movement/
│   ├── perception.py           # Computer vision and sensor processing
│   └── basic_agent.py         # Hybrid rule-based + ML agent
├── data/
│   ├── semantic/              # Semantic segmentation images
│   ├── rgb/                   # RGB camera images (optional)
│   └── metadata/              # GPS and steering data (JSON)
├── models/
│   ├── carla_dataset.py       # PyTorch dataset class
│   ├── bc_model.py            # Behavioral cloning model
│   └── train_bc.py            # Training script
├── evaluation/
│   └── evaluate_agent.py      # Comprehensive agent evaluation
├── utils/
│   ├── data_manager.py        # Dataset management and preprocessing
│   └── utils.py               # Utility functions
├── main_autonomous.py         # Main execution script
└── README.md                  # This file
```

## Installation

### Prerequisites
- CARLA 0.9.15+ (tested on 0.9.15)
- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies
```bash
pip install torch torchvision torchaudio
pip install opencv-python pygame numpy matplotlib
pip install pandas scikit-learn tqdm
pip install carla  # or add CARLA PythonAPI to your path
```

### CARLA Setup
1. Download CARLA 0.9.15 from [CARLA Releases](https://github.com/carla-simulator/carla/releases)
2. Extract and run the server:
```bash
cd CARLA_0.9.15/
./CarlaUE4.sh  # Linux
# or CarlaUE4.exe on Windows
```

## Quick Start

### 1. Data Collection
Collect training data using the autopilot:

```bash
python sensors/capture_sensors.py --duration 1800  # 30 minutes
```

This will collect:
- Semantic segmentation images (800x600)
- GPS coordinates 
- Steering/throttle/brake data
- Synchronized frame data

### 2. Data Management
Process and analyze your collected data:

```bash
python utils/data_manager.py --data_root ./data --action all
```

This will:
- Analyze dataset statistics
- Clean corrupted files
- Create train/val/test splits
- Balance steering distribution
- Generate visualizations
- Export training-ready data

### 3. Train ML Model
Train the behavioral cloning model:

```bash
python models/train_bc.py --data_dir ./data/training_ready --epochs 50
```

### 4. Run Autonomous Agent
Test your trained agent:

```bash
python main_autonomous.py --model_path ./models/bc_model.pth
```

### 5. Evaluate Performance
Comprehensive evaluation across multiple scenarios:

```bash
python evaluation/evaluate_agent.py
```

## System Architecture

### Hybrid Agent Design
```
Sensor Input → Perception → Planning → Safety Check → Control Output
     ↓           ↓           ↓           ↓            ↓
   RGB/Sem    Lane Det.   Path Plan   Rule Check   Steering
   LiDAR      Object Det.  Speed Plan  Collision    Throttle
   GPS        Traffic Det. Navigation  Traffic Laws  Brake
```

### Safety-First Approach
- **Hard Rules**: Traffic lights, collision avoidance, speed limits
- **ML Assistance**: Lane following, smooth control, obstacle navigation
- **Fail-Safe**: Emergency stop and rule override capabilities

## Usage Examples

### Basic Data Collection
```python
from sensors.capture_sensors import SensorManager

# Setup sensors and collect data
sensor_manager = SensorManager(vehicle, world)
sensor_manager.setup_sensors()

# Collect for 1000 frames
for i in range(1000):
    world.tick()
    data = sensor_manager.get_latest_data()
    sensor_manager.save_frame(data, i)
```

### Running Custom Agent
```python
from movement.basic_agent import HybridAgent
from movement.perception import PerceptionSystem

# Initialize systems
agent = HybridAgent(vehicle, world.get_map())
perception = PerceptionSystem()

# Main driving loop
while True:
    # Get sensor data
    sensor_data = get_sensor_data()
    
    # Process with perception system
    scene_data = perception.process_frame(
        sensor_data['rgb'],
        sensor_data['semantic']
    )
    
    # Get control command
    control = agent.run_step(scene_data)
    vehicle.apply_control(control)
```

### Custom Evaluation
```python
from evaluation.evaluate_agent import AgentEvaluator

evaluator = AgentEvaluator(client, world)
results = evaluator.run_evaluation(
    scenario='urban_driving',
    duration=300,
    traffic_density='high'
)

print(f"Safety Score: {results['scores']['safety']}/100")
print(f"Overall Score: {results['scores']['overall']}/100")
```

## Evaluation Metrics

### Safety Metrics
- Traffic light violations
- Speed limit violations  
- Collision count
- Lane departure incidents
- Emergency stops

### Performance Metrics
- Route completion rate
- Average speed
- Time to destination
- Fuel efficiency equivalent

### Comfort Metrics
- Steering smoothness
- Acceleration smoothness
- Passenger comfort score
- Following distance maintenance

## Configuration

### Key Parameters

**Data Collection** (`sensors/capture_sensors.py`):
```python
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
FRAME_RATE = 20  # fps
SENSOR_TICK = 0.05  # seconds
```

**Agent Behavior** (`movement/basic_agent.py`):
```python
SAFETY_DISTANCE = 3.0  # meters
MAX_SPEED = 50  # km/h
REACTION_TIME = 0.3  # seconds
EMERGENCY_BRAKE_DISTANCE = 5.0  # meters
```

**Training** (`models/train_bc.py`):
```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
IMAGE_SIZE = (224, 224)
```

## Troubleshooting

### Common Issues

**Vehicle Won't Move**
```python
# Add this after spawning vehicle
vehicle = world.spawn_actor(blueprint, spawn_point)
world.tick()  # Critical: let vehicle settle
vehicle.set_autopilot(True)
```

**Sensor Data Not Saving**
- Check disk space and permissions
- Verify CARLA server is running in synchronous mode
- Ensure output directories exist

**Training Convergence Issues**
- Check dataset balance (straight vs turning)
- Verify image preprocessing pipeline
- Monitor loss curves and learning rate

**Agent Evaluation Errors**
- Ensure all required sensor data is available
- Check CARLA version compatibility
- Verify traffic manager settings

### Performance Optimization

**For Better FPS**:
- Reduce camera resolution
- Disable unnecessary sensors
- Use semantic-only (no RGB) collection
- Run headless (no pygame display)

**For Better Training**:
- Balance dataset steering distribution
- Use data augmentation
- Implement curriculum learning
- Add more diverse scenarios

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CARLA Simulator team for the excellent autonomous driving platform
- PyTorch community for deep learning frameworks
- OpenCV community for computer vision tools

## References

- [CARLA Documentation](https://carla.readthedocs.io/)
- [Behavioral Cloning Paper](https://arxiv.org/abs/1604.07316)
- [End-to-End Deep Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)

---

**Happy Autonomous Driving!**