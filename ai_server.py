from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image
from google import genai
from dotenv import load_dotenv
import io
import os
import logging
from datetime import datetime
from infer import infer, visualize_annotations

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AI] %(message)s",
    handlers=[
        logging.FileHandler("ai_server.log"),
        logging.StreamHandler(),  # Also print to console
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_ID = "gemini-robotics-er-1.5-preview"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

logger.info(f"AI Server started at {datetime.now()}")
logger.info(f"Using model: {MODEL_ID}")


@app.post("/infer")
async def infer_endpoint(frame: UploadFile):
    logger.info(f"Received frame: {frame.filename}, content_type: {frame.content_type}")

    try:
        img_bytes = await frame.read()
        logger.info(f"Image size: {len(img_bytes)} bytes")

        # Load image for processing
        img = Image.open(io.BytesIO(img_bytes))

        # Convert to RGB if needed (handles RGBA, grayscale, etc.)
        if img.mode != "RGB":
            logger.info(f"Converting image from {img.mode} to RGB")
            img = img.convert("RGB")

        width, height = img.size
        logger.info(f"Image dimensions: {width}x{height}")

        # Define what to detect
        queries = ["glasses"]

        # Run inference
        logger.info(f"Calling Gemini API for: {queries}")
        annotations = infer(
            MODEL_ID, client, img_bytes, queries, mime_type="image/jpeg"
        )
        logger.info(f"Detected {len(annotations)} object(s): {annotations}")

        # Draw annotations on image
        annotated_img = visualize_annotations(img, annotations)
        logger.info(f"Annotations drawn on image")

        # Return processed image
        output_buffer = io.BytesIO()
        annotated_img.save(output_buffer, format="JPEG")
        processed_img_bytes = output_buffer.getvalue()
        logger.info(f"Returning processed image: {len(processed_img_bytes)} bytes")

        return Response(content=processed_img_bytes, media_type="image/jpeg")

    except Exception as e:
        logger.error(f"ERROR during inference: {type(e).__name__}: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

        # Return error image
        try:
            img = Image.new("RGB", (640, 480), color="black")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(img)
            draw.text((10, 10), f"Error: {str(e)[:100]}", fill="red")

            output_buffer = io.BytesIO()
            img.save(output_buffer, format="JPEG")
            return Response(content=output_buffer.getvalue(), media_type="image/jpeg")
        except:
            return Response(content=f"Error: {str(e)}", status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ai_server:app", host="0.0.0.0", port=5001, reload=True)
