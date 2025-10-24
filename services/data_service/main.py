"""
Data Service Main Entry Point
Imports and runs the production API
"""
from .api import app

# Re-export app for uvicorn
__all__ = ['app']
