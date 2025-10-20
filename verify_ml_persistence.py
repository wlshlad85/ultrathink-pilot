#!/usr/bin/env python3
"""
Verification script for ML Persistence setup.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required imports work."""
    print("Checking imports...")
    try:
        from ml_persistence import ExperimentTracker, ModelRegistry, DatasetManager, MetricsLogger
        print("✓ All ml_persistence imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_database():
    """Check if database exists and has correct schema."""
    print("\nChecking database...")
    db_path = Path("ml_experiments.db")
    
    if not db_path.exists():
        print("✗ Database not found - running initialization...")
        from ml_persistence.core import MLDatabase
        db = MLDatabase()
        print("✓ Database initialized")
    else:
        print("✓ Database exists")
    
    # Check tables
    import sqlite3
    conn = sqlite3.connect("ml_experiments.db")
    cursor = conn.cursor()
    
    expected_tables = [
        'experiments', 'models', 'datasets', 'metrics',
        'hyperparameters', 'artifacts', 'experiment_datasets'
    ]
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    missing_tables = [t for t in expected_tables if t not in tables]
    
    if missing_tables:
        print(f"✗ Missing tables: {missing_tables}")
        conn.close()
        return False
    
    print(f"✓ All {len(expected_tables)} tables exist")
    
    # Check views
    cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
    views = [row[0] for row in cursor.fetchall()]
    print(f"✓ Views created: {', '.join(views)}")
    
    conn.close()
    return True

def test_basic_operations():
    """Test basic create/read operations."""
    print("\nTesting basic operations...")
    
    try:
        from ml_persistence import ExperimentTracker
        
        tracker = ExperimentTracker()
        
        # Create a test experiment
        exp_id = tracker.start_experiment(
            name="Test Experiment",
            experiment_type="test",
            description="Verification test",
            tags=["test", "verification"]
        )
        print(f"✓ Created test experiment (ID: {exp_id})")
        
        # List experiments
        experiments = tracker.list_experiments(limit=5)
        print(f"✓ Listed {len(experiments)} experiments")
        
        # End experiment
        tracker.end_experiment(status="completed")
        print("✓ Completed test experiment")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during operations: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("ML Persistence Verification")
    print("=" * 60)
    print()
    
    checks = [
        check_imports(),
        check_database(),
        test_basic_operations()
    ]
    
    print()
    print("=" * 60)
    if all(checks):
        print("✅ All checks passed! ML Persistence is ready to use.")
        print()
        print("Quick Start:")
        print("  from ml_persistence import ExperimentTracker")
        print("  tracker = ExperimentTracker()")
        print("  exp_id = tracker.start_experiment(")
        print("      name='My Experiment',")
        print("      experiment_type='rl'")
        print("  )")
        print()
        print("See ml_persistence/README.md for more examples.")
        return 0
    else:
        print("❌ Some checks failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

