#!/usr/bin/env python3
"""
Run the FastAPI server for Conference Paper Agent.

Usage:
    python run_server.py
    
Or with uvicorn directly:
    uvicorn src.api.app:app --reload --port 8000
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment variables")
    print("Create a .env file with OPENAI_API_KEY=sk-...")
    sys.exit(1)


def main():
    """Run the API server"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print(f"""
╔══════════════════════════════════════════════════════╗
║       CONFERENCE PAPER AGENT API SERVER              ║
╠══════════════════════════════════════════════════════╣
║  Host: {host:<45} ║
║  Port: {port:<45} ║
║  Docs: http://{host}:{port}/docs{' ' * 25} ║
╚══════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
