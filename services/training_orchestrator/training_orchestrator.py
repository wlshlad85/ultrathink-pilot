"""
Training Orchestrator Service - Celery-based distributed ML training
Agent: ml-training-specialist

Coordinates model training, hyperparameter tuning, and experiment tracking.
Uses Celery for distributed task execution and MLflow for experiment management.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from celery import Celery, group, chord
from kafka import KafkaProducer
import json
import logging
from datetime import datetime
import redis
import pickle
import mlflow
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery with Redis broker
celery_app = Celery(
    'training_orchestrator',
    broker='redis://redis:6379/3',
    backend='redis://redis:6379/3'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50
)

# Redis client for data storage
redis_client = redis.Redis(host='redis', port=6379, db=3, decode_responses=False)

# Kafka producer for status updates
kafka_producer = KafkaProducer(
    bootstrap_servers='kafka-1:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

class SimpleStrategyModel(nn.Module):
    """Simple neural network for strategy prediction"""

    def __init__(self, input_dim=60, hidden_dim=128, output_dim=5):
        super(SimpleStrategyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

@celery_app.task(bind=True, name='training_orchestrator.fetch_training_data')
def fetch_training_data(self, symbol: str, start_date: str, end_date: str) -> str:
    """
    Fetch training data from Data Service and cache to Redis
    Returns cache key for the dataset
    """
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Fetching data...'})

        # Placeholder: In production, fetch from Data Service API
        # For now, generate synthetic data
        n_samples = 1000
        n_features = 60

        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 5, n_samples).astype(np.int64)

        # Cache dataset to Redis
        dataset_key = f"training_data:{symbol}:{datetime.utcnow().timestamp()}"
        dataset = {'X': X.tolist(), 'y': y.tolist()}
        redis_client.setex(dataset_key, 3600, pickle.dumps(dataset))

        logger.info(f"Cached training data: {dataset_key} ({n_samples} samples)")
        return dataset_key

    except Exception as e:
        logger.error(f"Data fetch error: {e}")
        raise

@celery_app.task(bind=True, name='training_orchestrator.preprocess_features')
def preprocess_features(self, dataset_key: str) -> str:
    """
    Preprocess features: normalization, feature selection, etc.
    Returns cache key for preprocessed dataset
    """
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Preprocessing features...'})

        # Load dataset from Redis
        dataset_bytes = redis_client.get(dataset_key)
        dataset = pickle.loads(dataset_bytes)

        X = np.array(dataset['X'])
        y = np.array(dataset['y'])

        # Normalize features (z-score)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_normalized = (X - X_mean) / X_std

        # Cache preprocessed data
        preprocessed_key = f"preprocessed:{dataset_key}"
        preprocessed_data = {
            'X': X_normalized.tolist(),
            'y': y.tolist(),
            'mean': X_mean.tolist(),
            'std': X_std.tolist()
        }
        redis_client.setex(preprocessed_key, 3600, pickle.dumps(preprocessed_data))

        logger.info(f"Preprocessed features: {preprocessed_key}")
        return preprocessed_key

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise

@celery_app.task(bind=True, name='training_orchestrator.train_model')
def train_model(self, preprocessed_key: str, hyperparams: Dict) -> Dict:
    """
    Train strategy selection model with given hyperparameters
    Returns training metrics and model cache key
    """
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Training model...'})

        # Load preprocessed data
        data_bytes = redis_client.get(preprocessed_key)
        data = pickle.loads(data_bytes)

        X = torch.FloatTensor(data['X'])
        y = torch.LongTensor(data['y'])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleStrategyModel(
            input_dim=X.shape[1],
            hidden_dim=hyperparams.get('hidden_dim', 128),
            output_dim=5
        ).to(device)

        # Training configuration
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.get('lr', 0.001))
        criterion = nn.CrossEntropyLoss()
        epochs = hyperparams.get('epochs', 50)
        batch_size = hyperparams.get('batch_size', 32)

        # MLflow experiment tracking
        mlflow.set_tracking_uri('http://mlflow:5000')
        mlflow.set_experiment('strategy_training')

        with mlflow.start_run():
            mlflow.log_params(hyperparams)

            # Training loop
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size].to(device)
                    batch_y = y_train[i:i+batch_size].to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / (len(X_train) / batch_size)

                # Validation
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_test.to(device))
                        val_loss = criterion(val_outputs, y_test.to(device))
                        val_acc = (val_outputs.argmax(1) == y_test.to(device)).float().mean()

                    mlflow.log_metrics({
                        'train_loss': avg_loss,
                        'val_loss': val_loss.item(),
                        'val_accuracy': val_acc.item()
                    }, step=epoch)

                    logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                    model.train()

            # Final evaluation
            model.eval()
            with torch.no_grad():
                final_outputs = model(X_test.to(device))
                final_loss = criterion(final_outputs, y_test.to(device))
                final_acc = (final_outputs.argmax(1) == y_test.to(device)).float().mean()

            # Cache trained model
            model_key = f"trained_model:{preprocessed_key}:{datetime.utcnow().timestamp()}"
            model_state = {
                'state_dict': model.state_dict(),
                'hyperparams': hyperparams,
                'metrics': {
                    'final_loss': final_loss.item(),
                    'final_accuracy': final_acc.item()
                }
            }
            redis_client.setex(model_key, 7200, pickle.dumps(model_state))

            # Log model to MLflow
            mlflow.pytorch.log_model(model, "model")

            metrics = {
                'final_loss': final_loss.item(),
                'final_accuracy': final_acc.item(),
                'model_key': model_key,
                'device': str(device)
            }

            logger.info(f"Training complete: {model_key} - Accuracy: {final_acc:.4f}")
            return metrics

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise

@celery_app.task(name='training_orchestrator.publish_training_status')
def publish_training_status(results: List[Dict], training_id: str):
    """
    Aggregate training results and publish status to Kafka
    """
    try:
        status = {
            'training_id': training_id,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'completed',
            'num_models': len(results),
            'best_accuracy': max(r['final_accuracy'] for r in results),
            'models': results
        }

        kafka_producer.send('training_status', value=status)
        logger.info(f"Published training status: {training_id}")
        return status

    except Exception as e:
        logger.error(f"Status publish error: {e}")
        raise

@celery_app.task(name='training_orchestrator.orchestrate_training')
def orchestrate_training(symbol: str, start_date: str, end_date: str,
                         hyperparams_list: List[Dict]) -> str:
    """
    Orchestrate full training pipeline with multiple hyperparameter configurations
    Uses Celery chord for parallel training
    """
    training_id = f"training_{symbol}_{datetime.utcnow().timestamp()}"

    logger.info(f"Starting orchestrated training: {training_id}")

    # Step 1: Fetch training data
    data_task = fetch_training_data.s(symbol, start_date, end_date)

    # Step 2: Preprocess features
    preprocess_task = preprocess_features.s()

    # Step 3: Train multiple models in parallel with different hyperparameters
    training_tasks = group([
        train_model.s(hyperparams) for hyperparams in hyperparams_list
    ])

    # Step 4: Aggregate results and publish status
    callback_task = publish_training_status.s(training_id)

    # Build pipeline: data -> preprocess -> parallel training -> publish
    pipeline = (data_task | preprocess_task | training_tasks | callback_task)
    result = pipeline.apply_async()

    logger.info(f"Training pipeline initiated: {training_id}")
    return training_id

def main():
    """Start Celery worker"""
    logger.info("Training Orchestrator Service starting...")

    # Test connection to Redis and Kafka
    try:
        redis_client.ping()
        logger.info("✅ Redis connection OK")
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")

    logger.info("Training Orchestrator ready for tasks")
    logger.info("Start worker with: celery -A training_orchestrator worker --loglevel=info")

if __name__ == '__main__':
    main()
