#!/usr/bin/env python3
"""Start the Local Multi-AI Agent server.

    python run.py
    python run.py --device cuda
    python run.py --device cpu
    python run.py --port 8000
"""

from app.cli import main

if __name__ == "__main__":
    main()
