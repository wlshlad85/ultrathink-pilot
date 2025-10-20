#!/bin/bash
# Setup script for ml_persistence

echo "Setting up ML Persistence..."
echo ""

# Check if numpy is installed
echo "Checking for numpy..."
if python3 -c "import numpy" 2>/dev/null; then
    echo "✓ numpy is installed"
else
    echo "✗ numpy not found - installing dependencies..."
    pip3 install -r requirements.txt
fi

echo ""
echo "Initializing ML database..."
python3 -m ml_persistence.core

echo ""
echo "Testing imports..."
python3 -c "
from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager, MetricsLogger
print('✓ All imports successful')
print('')
print('ML Persistence is ready to use!')
print('Example usage:')
print('  from ml_persistence import ExperimentTracker')
print('  tracker = ExperimentTracker()')
print('  exp_id = tracker.start_experiment(name=\"My Experiment\", experiment_type=\"rl\")')
"

echo ""
echo "Database created: ml_experiments.db"
echo "See ml_persistence/README.md for usage examples"

