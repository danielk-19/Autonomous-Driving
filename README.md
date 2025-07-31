# CARLA Autonomous Driving System

A hybrid machine learning + rule-based autonomous driving system for CARLA simulator with comprehensive data collection, training, and evaluation capabilities.

## Features

- **Hybrid Agent**: Combines ML perception with rule-based safety systems
- **Multi-Sensor Data Collection**: RGB, semantic segmentation, depth, GPS, IMU
- **Comprehensive Evaluation**: Safety, efficiency, and comfort metrics
- **Real-World Applicability**: Safety-first design with hard rule constraints
- **Data Management**: Complete pipeline from collection to training

## Project Structure

```
carla_autonomous_driving/
├── sensors/
│   └── capture_sensors.py      # Raw sensor data collection
├── movement/
│   ├── perception.py           # Computer vision processing
│   └── basic_agent.py         # Hybrid ML + rule-based agent
├── data/
│   ├── raw_sessions/          # Raw collected data
│   └── processed/             # ML-ready processed data
├── models/
│   ├── carla_dataset.py       # PyTorch dataset class
│   ├── bc_model.py            # Neural network architecture
│   └── train_bc.py            # Model training script
├── evaluation/
│   └── evaluate_agent.py      # Agent performance testing
├── utils/
│   ├── data_processor.py      # Raw → processed conversion
│   └── utils.py               # Utility functions
├── logs/                      # Runtime logs and metrics
├── main_autonomous.py         # Main execution script
└── README.md                  # Project documentation
```

## Installation

### Prerequisites

- CARLA 0.9.15+ (tested on 0.9.15)
- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies

```bash
pip install torch>=1.9.0 torchvision>=0.10.0 torchaudio
pip install opencv-python>=4.5.0 pygame>=2.0.0 numpy>=1.21.0
pip install pandas>=1.3.0 scikit-learn>=1.0.0 matplotlib>=3.4.0
pip install albumentations>=1.1.0
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
python sensors/capture_sensors.py --max-frames 5000 --autopilot
```

This will collect synchronized data at 20 FPS:

- RGB camera images (800x600)
- Semantic segmentation images (800x600)
- Depth maps
- GPS coordinates and IMU data
- Vehicle control commands (steering/throttle/brake)

Data is saved in session folders: `data/raw_sessions/session_YYYYMMDD_HHMMSS/`

### 2. Data Processing

Convert raw sessions to ML-ready format:

```bash
python utils/data_processor.py --data_root ./data --action process
```

This will:

- Combine RGB and semantic data into training images
- Consolidate metadata into episode-based measurements.json files
- Create balanced train/val/test splits
- Validate data quality and remove corrupted frames

**Alternative commands:**

```bash
# Process all sessions and analyze
python utils/data_processor.py --data_root ./data --action all

# Only analyze existing processed data
python utils/data_processor.py --data_root ./data --action analyze

# Validate processed data integrity
python utils/data_processor.py --data_root ./data --action validate

# Clean up corrupted episodes
python utils/data_processor.py --data_root ./data --action clean
```

### 3. Train ML Model

Train the behavioral cloning model:

```bash
python models/train_bc.py --data_dir ./data/processed --num_epochs 50 --batch_size 32
```

Model architecture:

- Input: RGB images (3, 224, 224)
- Output: [steering, throttle, brake]
- Backbone: ResNet18-based CNN
- Loss: Weighted MSE (steering weight = 2.0)

**Additional training options:**

```bash
# With custom learning rate and validation split
python models/train_bc.py --data_dir ./data/processed --num_epochs 100 --batch_size 64 --learning_rate 0.0001 --val_split 0.15

# Resume training from checkpoint
python models/train_bc.py --data_dir ./data/processed --resume ./models/checkpoint_epoch_25.pth
```

### 4. Run Autonomous Agent

Test your trained agent:

```bash
python main_autonomous.py --model ./checkpoints/final_model.pth --spawn-point 0
```

**Alternative run commands:**

```bash
# Auto-detect latest model with random spawn
python main_autonomous.py

# Headless mode for performance testing
python main_autonomous.py --model ./checkpoints/final_model.pth --no-display --frames 5000

# Specific route with traffic
python main_autonomous.py --model ./checkpoints/final_model.pth --spawn-point 5 --dest 20

# No data collection, display only
python main_autonomous.py --model ./checkpoints/final_model.pth --no-save

# Custom CARLA server
python main_autonomous.py --host 192.168.1.100 --port 2000 --model ./checkpoints/final_model.pth
```

### 5. Evaluate Performance

Comprehensive evaluation across multiple scenarios:

```bash
python evaluation/evaluate_agent.py
```

The evaluation script will:

- Auto-detect the latest trained model
- Run tests on multiple scenarios (urban, highway, traffic lights, intersections)
- Generate detailed performance reports
- Save results to `evaluation_results/`

## Data Structure

### Raw Data Collection Format

```
data/raw_sessions/session_YYYYMMDD_HHMMSS/
├── rgb/           # RGB images: 000000.png, 000001.png, ...
├── semantic/      # Semantic images: 000000.png, 000001.png, ...
├── depth/         # Depth maps: 000000.npy, 000001.npy, ...
├── gps/           # GPS data: 000000.json, 000001.json, ...
├── imu/           # IMU data: 000000.json, 000001.json, ...
└── control/       # Control data: 000000.json, 000001.json, ...
```

### Processed Data Format (ML-ready)

```
data/processed/
├── episodes/
│   ├── episode_001/
│   │   ├── images/           # Combined training images: 000000.png, ...
│   │   └── measurements.json # All frame metadata in single file
│   └── episode_002/
│       ├── images/
│       └── measurements.json
├── train_samples.json
├── val_samples.json
├── test_samples.json
└── data_splits.json
```

## System Architecture

### Data Flow Pipeline

```
1. Collection:  capture_sensors.py → Raw session data
2. Processing:  data_processor.py → Processed episodes
3. Training:    train_bc.py + dataset + model → Trained weights
4. Deployment:  main_autonomous.py + agent → Autonomous driving
5. Evaluation:  evaluate_agent.py → Performance metrics
```

### Hybrid Agent Design

```
Sensor Input → Perception → ML Prediction → Safety Check → Control Output
     ↓           ↓              ↓             ↓            ↓
   RGB/Sem    Lane Det.     Steering      Rule Check   Final Steering
   Depth      Object Det.   Throttle      Collision    Final Throttle
   GPS/IMU    Traffic Det.  Brake         Traffic Laws Final Brake
```

### Safety-First Approach (Non-negotiable Rules)

- **Emergency brake** on collision risk
- **Traffic light compliance** - mandatory stops
- **Speed limit enforcement** (50 km/h maximum)
- **Lane departure prevention**
- **Following distance maintenance** (3m minimum)
- **ML override capability** for safety violations

## Configuration

### Key Parameters

**Data Collection** (`sensors/capture_sensors.py`):

```python
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
FRAME_RATE = 20  # fps
SENSOR_TICK = 0.05  # seconds
FOV = 90  # degrees
```

**Model Training** (`models/train_bc.py`):

```python
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
IMAGE_SIZE = (224, 224)
STEERING_WEIGHT = 2.0  # Loss weighting
```

**Agent Safety** (`movement/basic_agent.py`):

```python
SAFETY_DISTANCE = 3.0  # meters
MAX_SPEED = 50  # km/h
REACTION_TIME = 0.3  # seconds
EMERGENCY_BRAKE_DISTANCE = 5.0  # meters
```

## Performance Targets

- **Training**: <0.1 MSE on validation set
- **Safety**: <1 violation per 1000 frames
- **Efficiency**: >80% route completion rate
- **Comfort**: <0.5 m/s² average acceleration

## Command Line Arguments Reference

### Data Processor (`utils/data_processor.py`)

```bash
--data_root     # Path to dataset root directory (required)
--action        # Action: process, analyze, validate, clean, all (default: all)
--log_level     # Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
```

### Model Training (`models/train_bc.py`)

```bash
--data_root     # Path to dataset root directory (required)
--epochs        # Number of training epochs (default: 50)
--batch_size    # Training batch size (default: 32)
--lr            # Learning rate (default: 0.001)
--val_split     # Validation split ratio (default: 0.1)
--resume        # Resume from checkpoint path
--save_freq     # Save checkpoint frequency (default: 10)
--log_level     # Logging level (default: INFO)
```

### Main Autonomous (`main_autonomous.py`)

```bash
--host          # CARLA server IP (default: localhost)
--port          # CARLA server port (default: 2000)
--model         # Path to trained model (auto-detects if not specified)
--no-save       # Disable data collection
--frames        # Max frames to run (default: 10000)
--spawn-point   # Vehicle spawn point index
--dest          # Destination spawn point index
--no-traffic    # Disable background traffic
--no-display    # Run headless (no visualization)
```

### Evaluation (`evaluation/evaluate_agent.py`)

The evaluation script currently uses default settings and auto-detection. No command line arguments are required.

## Evaluation Metrics

### Safety Metrics

- Traffic light violations
- Speed limit violations
- Collision count
- Lane departure incidents
- Emergency stops triggered

### Performance Metrics

- Route completion rate
- Average speed maintenance
- Time to destination
- Navigation accuracy

### Comfort Metrics

- Steering smoothness (jerk minimization)
- Acceleration smoothness
- Passenger comfort score
- Following distance consistency

## Usage Examples

### Basic Data Collection

```python
from sensors.capture_sensors import SensorManager

# Initialize and collect data
sensor_manager = SensorManager(vehicle, world)
sensor_manager.setup_sensors()

# Collect synchronized frames
for frame_id in range(1000):
    world.tick()
    data = sensor_manager.get_latest_data()
    sensor_manager.save_frame(data, frame_id)
```

### Running the Autonomous Agent

```python
from movement.basic_agent import HybridAgent
from models.bc_model import BehavioralCloningModel

# Load trained model
model = BehavioralCloningModel()
model.load_state_dict(torch.load('models/bc_model.pth'))

# Initialize hybrid agent
agent = HybridAgent(vehicle, world.get_map(), model)

# Main driving loop
while True:
    sensor_data = get_sensor_data()
    control = agent.run_step(sensor_data)
    vehicle.apply_control(control)
```

### Custom Evaluation

```python
from evaluation.evaluate_agent import AgentEvaluator

evaluator = AgentEvaluator(client, world)
results = evaluator.run_evaluation(
    model_path='models/bc_model.pth',
    scenarios=['urban_driving', 'highway', 'rural'],
    duration=300  # seconds per scenario
)

print(f"Safety Score: {results['safety_score']}/100")
print(f"Overall Performance: {results['overall_score']}/100")
```

## Pipeline Verification

To verify your complete pipeline works:

```bash
# 1. Collect small dataset
python sensors/capture_sensors.py --max-frames 1000 --autopilot

# 2. Process the data
python utils/data_processor.py --data_root ./data --action all

# 3. Train for few epochs
python models/train_bc.py --data_root ./data --epochs 5 --batch_size 16

# 4. Test the agent
python main_autonomous.py --frames 500 --no-traffic

# 5. Evaluate performance
python evaluation/evaluate_agent.py
```

## Troubleshooting

### Common Issues

**Data Collection Problems**

- Ensure CARLA is running in synchronous mode
- Check disk space for large datasets
- Verify sensor tick rate matches world tick rate
- Use `world.tick()` after vehicle spawn

**Training Convergence Issues**

- Balance steering distribution in dataset
- Check for corrupted images or metadata
- Monitor loss curves and adjust learning rate
- Ensure proper train/val/test splits

**Agent Performance Issues**

- Verify model input preprocessing matches training
- Check safety rule implementation
- Monitor sensor data quality in real-time
- Validate traffic light detection accuracy

### Performance Optimization

**For Data Collection**:

- Use semantic-only mode to save storage
- Reduce camera resolution for faster processing
- Run headless (no pygame display)
- Use SSD storage for better I/O performance

**For Training**:

- Use data augmentation (rotation, brightness)
- Implement curriculum learning
- Use mixed precision training
- Balance dataset across scenarios

**For Runtime**:

- Optimize model inference with TensorRT
- Use efficient image preprocessing
- Minimize sensor data copying
- Implement multi-threading for sensors

### Common Fixes

**Data Processor Issues**

```bash
# If processing fails, try cleaning first
python utils/data_processor.py --data_root ./data --action clean

# Then reprocess
python utils/data_processor.py --data_root ./data --action process
```

**Training Issues**

```bash
# Check data integrity before training
python utils/data_processor.py --data_root ./data --action validate

# Use smaller batch size if memory issues
python models/train_bc.py --data_root ./data --batch_size 16
```

**Runtime Issues**

```bash
# Run without display if performance issues
python main_autonomous.py --no-display --no-traffic

# Use fallback control if model fails
python main_autonomous.py --model none
```

## File Naming Conventions

- **Images**: `{frame_id:06d}.png` (e.g., `000000.png`)
- **Raw metadata**: `{frame_id:06d}.json` per sensor type
- **Processed episodes**: `episode_{id:03d}` (e.g., `episode_001`)
- **Sessions**: `session_{timestamp}` (e.g., `session_20240101_120000`)

## System Requirements

### Minimum Requirements

- CPU: Intel i5-8400 or AMD Ryzen 5 2600
- RAM: 16 GB
- GPU: NVIDIA GTX 1060 6GB or AMD RX 580 8GB
- Storage: 100 GB available space
- OS: Ubuntu 18.04+ or Windows 10

### Recommended Requirements

- CPU: Intel i7-10700K or AMD Ryzen 7 3700X
- RAM: 32 GB
- GPU: NVIDIA RTX 3070 or better
- Storage: 500 GB SSD
- OS: Ubuntu 20.04+ or Windows 11

## Interactive Controls

When running the autonomous system with display mode:

- **ESC** - Exit system
- **R** - Set new random destination
- **S** - Toggle data collection
- **T** - Spawn additional traffic
- **H** - Print agent health status

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the specifications
4. Add unit tests for new components
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Testing Requirements

- Unit tests for each major component
- Integration tests for data pipeline
- Performance benchmarks for agent
- Memory and speed profiling
- Safety scenario validation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CARLA Simulator team for the autonomous driving platform
- PyTorch community for deep learning frameworks
- OpenCV community for computer vision tools

## References

- [CARLA Documentation](https://carla.readthedocs.io/)
- [End-to-End Deep Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [Dynamic Conditional Imitation Learning for Autonomous Driving](https://arxiv.org/abs/2211.11579)
- [Autonomous Driving with Deep Reinforcement Learning in CARLA Simulation](https://arxiv.org/abs/2306.11217)

---

**Happy Autonomous Driving!**
