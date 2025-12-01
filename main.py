from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import requests
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MAIN] %(message)s",
    handlers=[
        logging.FileHandler("main_server.log"),
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AI_SERVER_URL = "http://localhost:5001/infer"

logger.info(f"Main Server started at {datetime.now()}")
logger.info(f"AI Server URL: {AI_SERVER_URL}")


@app.post("/frames")
async def receive_frame(frame: UploadFile):
    logger.info(f"Received frame from browser: {frame.filename}")

    try:
        image_bytes = await frame.read()
        logger.info(f"Frame size: {len(image_bytes)} bytes")

        # AI 서버로 프레임 전달
        logger.info(f"Forwarding to AI server at {AI_SERVER_URL}")
        ai_response = requests.post(
            AI_SERVER_URL,
            files={"frame": ("frame.jpg", image_bytes, "image/jpeg")},
            timeout=30,  # 30 second timeout
        )

        logger.info(f"AI server response status: {ai_response.status_code}")

        if ai_response.status_code == 200:
            logger.info(f"AI server returned {len(ai_response.content)} bytes")
            # AI 서버가 이미지(jpg)를 반환 → 그대로 클라이언트로 전송
            return Response(content=ai_response.content, media_type="image/jpeg")
        else:
            logger.error(
                f"AI server error: {ai_response.status_code} - {ai_response.text}"
            )
            # Return original image if AI server fails
            return Response(content=image_bytes, media_type="image/jpeg")

    except requests.exceptions.Timeout:
        logger.error("ERROR: AI server timeout")
        return Response(content=image_bytes, media_type="image/jpeg")
    except requests.exceptions.ConnectionError:
        logger.error("ERROR: Cannot connect to AI server - is it running on port 5001?")
        return Response(content=image_bytes, media_type="image/jpeg")
    except Exception as e:
        logger.error(f"ERROR: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return Response(content=image_bytes, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=5002, reload=True)
