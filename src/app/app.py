"""
App configuration and optional Gradio UI entry.
Bank churn: re-export app from main for uvicorn src.app.app:app compatibility.
"""
from src.app.main import app

__all__ = ["app"]
