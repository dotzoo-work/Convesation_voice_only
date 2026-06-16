from fastapi.staticfiles import StaticFiles
import os
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, LiveKitAPI
from dotenv import load_dotenv
import logging

logger = logging.getLogger("api")

# Load environment variables from the correct path
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.local'))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/livekit/token")
def get_livekit_token(
    room: str = Query(...),
    identity: str = Query(...)
):
    try:
        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        livekit_url = os.getenv("LIVEKIT_URL")
        
        if not api_key or not api_secret:
            return {"error": "LiveKit credentials not configured"}
        
        import jwt
        import time
        
        # Create JWT payload manually with audio permissions
        payload = {
            "iss": api_key,
            "sub": identity,
            "name": identity,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,  # 1 hour
            "video": {
                "room": room,
                "roomJoin": True,
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,
                "canUpdateOwnMetadata": True
            }
        }
        
        token = jwt.encode(payload, api_secret, algorithm="HS256")
        
        return {
            "token": token,
            "url": livekit_url,
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/livekit/dispatch-agent")
async def dispatch_agent(request: Request):
    try:
        import json as _json
        data = await request.json()
        room_name = data.get("room")
        lang = data.get("lang", "en")

        api_key = os.getenv("LIVEKIT_API_KEY")
        api_secret = os.getenv("LIVEKIT_API_SECRET")
        livekit_url = os.getenv("LIVEKIT_URL")

        lkapi = LiveKitAPI(livekit_url, api_key, api_secret)

        try:
            from livekit.api import CreateRoomRequest
            await lkapi.room.create_room(CreateRoomRequest(
                name=room_name,
                metadata=_json.dumps({"lang": lang})
            ))
            logger.info(f"🚀 Room created: {room_name} | lang: {lang}")
        except Exception as room_error:
            logger.info(f"Room may already exist: {room_error}")

        await lkapi.aclose()
        return {"status": "dispatched", "room": room_name}
    except Exception as e:
        logger.error(f"Dispatch error: {e}")
        return {"error": str(e)}


_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend_build")
if os.path.isdir(_frontend_dir):
    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")