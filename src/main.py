import asyncio
import os
import uvicorn
import logging
from multiprocessing import Process
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.local'))

from agent_server import server
from api import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


def run_api():
    uvicorn.run(app, host="0.0.0.0", port=8002)


def run_agent():
    import sys
    sys.argv = ["agent", "dev"]
    logger.info("🔥 Starting agent server...")
    logger.info(f"LiveKit URL: {os.getenv('LIVEKIT_URL')}")
    from livekit.agents import cli
    cli.run_app(server)


if __name__ == "__main__":
    Process(target=run_api).start()
    run_agent()


    ## run -> python main.py
    ## for copy out fronetdn build  run -> xcopy /E /Y /I frontend\out agent-starter-python\src\frontend_build
