#!/bin/bash
# Install dependencies for ml_persistence in current conda environment

echo "Installing numpy and pandas in current environment..."
echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Install just the missing dependencies
pip install numpy pandas

echo ""
echo "Verifying installation..."
python3 -c "import numpy; import pandas; print('✓ numpy version:', numpy.__version__); print('✓ pandas version:', pandas.__version__)"

echo ""
echo "Done! Now run: python3 verify_ml_persistence.py"

