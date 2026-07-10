"""Pytest configuration - ensures the backend root is importable
so rom app.main import app works regardless of how pytest is invoked."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
