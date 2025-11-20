# Conformal DOA Tracking

Implementation of Conformal Prediction-Aware Direction-of-Arrival (DOA) Tracking using U-Net architecture. This repository contains data generation, training code, and Docker environments for acoustic source localization with uncertainty quantification.

## 🏗️ Repository Structure

```
├── data_generation/          # Acoustic data simulation pipeline
│   ├── main.py              # Main data generation script
│   ├── scene_gen.py         # Room and source scenario generation
│   ├── rir_gen.py           # Room impulse response generation
│   ├── processing.py        # Audio signal processing
│   └── signal_generator_python/  # C++ signal generator bindings
├── UNet_tracking/           # U-Net training and testing
│   ├── train_model.py       # Model training script
│   ├── test_model.py        # Model testing and evaluation
│   ├── unet/               # U-Net model architecture
│   └── utils/              # Utilities for training and conformal prediction
└── docker/                  # Docker environments for reproducibility
    ├── unet/               # Training environment
    ├── dataset_gen/        # Data generation environment
    └── README.md           # Docker setup instructions
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch with CUDA support
- Docker with NVIDIA Container Runtime (optional)

### Option 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/MayaVB/conformal-doa-tracking.git
cd conformal-doa-tracking

# Build Docker images
docker-compose build

# Generate data
docker-compose run dataset_gen

# Train model
docker-compose run unet
```

### Option 2: Local Installation

```bash
# Install dependencies
pip install -r docker/unet/requirements.txt
pip install -r docker/dataset_gen/requirements.txt

# Setup wandb (for experiment tracking)
wandb login
```

## 📊 Data Generation

The data generation pipeline simulates acoustic scenarios with moving speakers in reverberant rooms using the AudioLabs signal generator with Python bindings.

### Prerequisites

Before generating data, you must build the C++ signal generator:

```bash
cd data_generation/signal_generator_python
./build_and_copy.sh
cd ..
```

This builds the AudioLabs signal generator with Python bindings for high-performance audio signal generation.

### Basic Usage

```bash
cd data_generation
python main.py \
  --output_folder /path/to/output \
  --num_scenes 100 \
  --clean_speech_dir /path/to/wsj0/dataset
```

**Required:** You must provide a path to clean speech utterances (e.g., WSJ0, LibriSpeech, or similar dataset) via `--clean_speech_dir`.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output_folder` | Directory where generated data will be saved | Required |
| `--clean_speech_dir` | Path to clean speech dataset (WSJ0, LibriSpeech, etc.) | Required |
| `--num_scenes` | Number of acoustic scenarios to generate | 5 |
| `--mics_num` | Number of microphones in array | 5 |
| `--T60_options` | Reverberation time options [0.2, 0.5, 0.8] | [0.2] |
| `--snr` | Signal-to-noise ratio (dB) | 20 |
| `--room_len_x_min/max` | Room x-dimension range (m) | 4-7 |
| `--aspect_ratio_min/max` | Room aspect ratio range | 1-1.5 |

### Speech Datasets

The data generation requires clean speech utterances. **Tested and verified with:**
- **WSJ0** (Wall Street Journal): Professional speech corpus - **Recommended**

Other datasets may work but have not been tested:
- **LibriSpeech**: Open-source audiobook recordings  
- **TIMIT**: Phonetically rich speech corpus
- **VCTK**: Multi-speaker English corpus

Download WSJ0 from LDC (Linguistic Data Consortium) and provide the path via `--clean_speech_dir`.


## 🎯 Model Training

Train the U-Net model for DOA estimation with conformal prediction capabilities.

### Basic Training

```bash
cd UNet_tracking
python train_model.py --rtf-estimation-method iRTF  --batch-size 30 --learning-rate 1e-4 --epochs 140 --save-checkpoint True
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epochs` | Number of training epochs | 140 |
| `--batch-size` | Training batch size | 2 |
| `--learning-rate` | Learning rate | 1e-4 |
| `--classes` | Number of DOA classes | 32 |
| `--features` | Feature type | 'SinCosWithoutSpec' |
| `--doa-grid-res` | DOA grid resolution (degrees) | 5 |

### Feature Options

- `ReImWithoutSpec`: Real/imaginary parts without spectrogram
- `ReImWithSpec`: Real/imaginary parts with spectrogram  
- `SinCosWithoutSpec`: Sin/cos features without spectrogram
- `SinCosWithSpec`: Sin/cos features with spectrogram

## 🧪 Model Testing & Evaluation

Test the trained model with conformal prediction for uncertainty quantification.

### Basic Testing

```bash
cd UNet_tracking
python test_model.py \
  --load ./checkpoints/checkpoint_epoch100.pth \
  --test-data-path /path/to/test/data
```

### Key Testing Parameters

| Parameter | Description |
|-----------|-------------|
| `--load` | Path to trained model checkpoint |
| `--test-data-path` | Path to test dataset |
| `--conformal` | Enable conformal prediction |
| `--calibration-data-path` | Path to calibration dataset |
| `--confidence-level` | Confidence level for prediction intervals |

### Conformal Prediction Example

```bash
python test_model.py \
  --load ./checkpoints/best_model.pth \
  --test-data-path ./test_data \
  --conformal \
  --calibration-data-path ./calibration_data \
  --confidence-level 0.9 \
  --output-dir ./results
```

## 📈 Results and Visualization

The testing script generates:
- DOA estimation accuracy metrics
- Conformal prediction interval coverage
- Error vs. true DOA plots
- Confidence interval visualizations
- CSV files with detailed results

Results are automatically logged to Weights & Biases (wandb) for experiment tracking.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{veisman2025conformal,
  title={Conformal Prediction-Aware DOA Tracking},
  author={Maya Veisman Barness},
  journal={Your Journal},
  year={2025}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AudioLabs** for the high-performance signal generator
- **Yoav Ellinson** and **Ilai Zaidel** for creating the Python wrapper for the AudioLabs signal generator
- NVIDIA for PyTorch containers
- The acoustic signal processing community
- **Prof. Sharon Gannot** and **Dr. Bracha Goldstein Laufer** for their guidance and contributions to the research
