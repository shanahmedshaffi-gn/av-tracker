#!/bin/bash

# PyAnnote Multi-Modal Speaker Recognition Environment Setup
# ==========================================================
# 
# This script sets up a complete development environment for the
# PyAnnote multi-modal speaker recognition system.
#
# Features:
# - Creates conda environment
# - Installs all required dependencies
# - Sets up development tools
# - Validates installation
# - Provides usage instructions
#
# Author: DevAgent Collaborative Development
# Created: September 8, 2025

set -euo pipefail

# Color definitions
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Configuration
ENVIRONMENT_NAME="pyannote-speaker-recognition"
PYTHON_VERSION="3.11"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Functions
log() {
    echo -e "${GREEN}[SETUP]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

check_conda() {
    if ! command -v conda &> /dev/null; then
        error "Conda is not installed or not in PATH"
        echo "Please install Miniconda or Anaconda first:"
        echo "https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    log "Conda found: $(conda --version)"
}

check_environment_exists() {
    if conda env list | grep -q "^${ENVIRONMENT_NAME}"; then
        return 0
    else
        return 1
    fi
}

create_environment() {
    log "Creating conda environment: ${ENVIRONMENT_NAME}"
    conda create -n "${ENVIRONMENT_NAME}" python="${PYTHON_VERSION}" -y
    log "Environment created successfully"
}

activate_environment() {
    log "Activating environment: ${ENVIRONMENT_NAME}"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENVIRONMENT_NAME}"
    log "Environment activated"
}

install_pytorch() {
    log "Installing PyTorch with optimal configuration..."
    
    # Detect platform and install appropriate PyTorch
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use MPS acceleration for Apple Silicon
        log "Detected macOS - installing PyTorch with MPS support"
        pip install torch torchaudio torchvision
    elif command -v nvidia-smi &> /dev/null; then
        # NVIDIA GPU detected
        log "Detected NVIDIA GPU - installing PyTorch with CUDA support"
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        # CPU only
        log "Installing CPU-only PyTorch"
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
}

install_audio_dependencies() {
    log "Installing audio processing dependencies..."
    
    # Install system audio dependencies first (if needed)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log "Note: On Linux, you may need to install system audio libraries:"
        log "  sudo apt-get install portaudio19-dev python3-pyaudio"
        log "  sudo apt-get install libasound-dev"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            log "Installing audio dependencies via Homebrew..."
            brew install portaudio || warn "Failed to install portaudio via brew (may already be installed)"
        else
            warn "Homebrew not found. You may need to install PortAudio manually."
        fi
    fi
}

install_requirements() {
    log "Installing Python requirements..."
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install core requirements
    if [[ -f "${PROJECT_DIR}/requirements.txt" ]]; then
        log "Installing core requirements..."
        pip install -r "${PROJECT_DIR}/requirements.txt"
    else
        error "requirements.txt not found in ${PROJECT_DIR}"
        exit 1
    fi
}

install_dev_requirements() {
    log "Installing development requirements..."
    
    if [[ -f "${PROJECT_DIR}/requirements-dev.txt" ]]; then
        pip install -r "${PROJECT_DIR}/requirements-dev.txt"
    else
        warn "requirements-dev.txt not found - skipping development dependencies"
    fi
}

install_chain_framework() {
    log "Checking for Chain framework..."
    
    # Try to install Chain if available
    pip install chain-ml 2>/dev/null || {
        warn "Chain framework not available via pip"
        log "You may need to install it from source if using chain-based implementations"
        log "See project documentation for Chain installation instructions"
    }
}

validate_installation() {
    log "Validating installation..."
    
    # Test core imports
    python -c "
import torch
import numpy as np
import librosa
import sounddevice
print('✅ Core dependencies imported successfully')

# Test PyTorch device availability
if torch.cuda.is_available():
    print('✅ CUDA available:', torch.cuda.get_device_name(0))
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS (Apple Silicon) available')
else:
    print('✅ CPU-only PyTorch (no GPU acceleration)')

# Test PyAnnote
try:
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    print('✅ PyAnnote.audio available')
except ImportError as e:
    print(f'❌ PyAnnote.audio import failed: {e}')

# Test SpeechBrain
try:
    import speechbrain
    print('✅ SpeechBrain available')
except ImportError as e:
    print(f'❌ SpeechBrain import failed: {e}')

# Test Whisper
try:
    import whisper
    print('✅ OpenAI Whisper available')
except ImportError as e:
    print(f'❌ Whisper import failed: {e}')

print('\\n🎉 Installation validation complete!')
"
}

create_activation_script() {
    log "Creating environment activation script..."
    
    cat > "${PROJECT_DIR}/activate_env.sh" << 'EOF'
#!/bin/bash
# Quick environment activation script
# Usage: source ./activate_env.sh

ENVIRONMENT_NAME="pyannote-speaker-recognition"

if conda env list | grep -q "^${ENVIRONMENT_NAME}"; then
    echo "Activating ${ENVIRONMENT_NAME} environment..."
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${ENVIRONMENT_NAME}"
    echo "✅ Environment activated"
    echo "📍 Project directory: $(pwd)"
    echo "🐍 Python: $(which python)"
    echo ""
    echo "Available scripts:"
    echo "  python audio_profile_creator.py    # Create speaker profiles"
    echo "  python test_audio_profile_creator.py  # Run tests"
    echo ""
else
    echo "❌ Environment ${ENVIRONMENT_NAME} not found"
    echo "Run ./setup_environment.sh to create it"
    exit 1
fi
EOF
    
    chmod +x "${PROJECT_DIR}/activate_env.sh"
    log "Created activation script: ./activate_env.sh"
}

show_usage_instructions() {
    log "Setup complete! 🎉"
    echo ""
    echo -e "${BLUE}=== Usage Instructions ===${NC}"
    echo ""
    echo "1. Activate the environment:"
    echo "   source ./activate_env.sh"
    echo ""
    echo "2. Create your first speaker profile:"
    echo "   python audio_profile_creator.py"
    echo ""
    echo "3. Run tests to verify everything works:"
    echo "   python test_audio_profile_creator.py"
    echo ""
    echo "4. Explore existing implementations:"
    echo "   ls *.py | grep -E '(all_|verification_|multi_)'"
    echo ""
    echo -e "${BLUE}=== Development Workflow ===${NC}"
    echo ""
    echo "• Use 'conda activate ${ENVIRONMENT_NAME}' to activate"
    echo "• Install new packages with 'pip install <package>'"
    echo "• Update requirements.txt when adding dependencies"
    echo "• Run 'conda deactivate' when done"
    echo ""
    echo -e "${BLUE}=== Project Structure ===${NC}"
    echo ""
    echo "📁 embeddings/     - Speaker profile embeddings"
    echo "📁 outputs/        - Recorded audio files"
    echo "📁 transcripts/    - Generated transcriptions"
    echo "📁 project/        - DevAgent project management"
    echo "🐍 audio_profile_creator.py - Enhanced profile creation"
    echo "🧪 test_*.py       - Test suites"
    echo "📄 requirements*.txt - Dependency specifications"
    echo ""
}

main() {
    echo -e "${BLUE}PyAnnote Multi-Modal Speaker Recognition Setup${NC}"
    echo -e "${BLUE}===============================================${NC}"
    echo ""
    
    # Pre-flight checks
    check_conda
    
    # Handle existing environment
    if check_environment_exists; then
        warn "Environment ${ENVIRONMENT_NAME} already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log "Removing existing environment..."
            conda env remove -n "${ENVIRONMENT_NAME}" -y
        else
            log "Using existing environment"
        fi
    fi
    
    # Create or use environment
    if ! check_environment_exists; then
        create_environment
    fi
    
    # Activate and install
    activate_environment
    install_audio_dependencies
    install_pytorch
    install_requirements
    install_dev_requirements
    install_chain_framework
    
    # Validate and finalize
    validate_installation
    create_activation_script
    show_usage_instructions
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "validate")
        check_conda
        activate_environment
        validate_installation
        ;;
    "clean")
        log "Removing environment: ${ENVIRONMENT_NAME}"
        conda env remove -n "${ENVIRONMENT_NAME}" -y || warn "Environment not found"
        rm -f "${PROJECT_DIR}/activate_env.sh"
        log "Cleanup complete"
        ;;
    *)
        echo "Usage: $0 [setup|validate|clean]"
        echo "  setup    - Create and configure environment (default)"
        echo "  validate - Test existing environment"
        echo "  clean    - Remove environment and files"
        exit 1
        ;;
esac
