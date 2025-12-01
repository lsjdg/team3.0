from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
from PIL import Image, ImageDraw, ImageFont


load_dotenv()


def infer(model, client, image_bytes, queries, mime_type="image/jpeg"):
    """
    Run inference on image to detect objects.

    Args:
        model: Model ID string
        client: Gemini client instance
        image_bytes: Raw image bytes
        queries: List of objects to detect (e.g., ["safety helmet", "hard hat"])
        mime_type: Image MIME type (default: "image/jpeg")

    Returns:
        List of annotations in format [{"point": [y, x], "label": "..."}, ...]
    """
    prompt = f"""
  Get all points matching the following objects: {', '.join(queries)}. The
  label returned should be an identifying name for the object detected.
  The answer should follow the json format:

  [{{"point": [y, x], "label": "helmet"}}, ...]. The points are in

  [y, x] format normalized to 0-1000.
  If no objects are detected, return an empty array [].
  """

    image_response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.3, thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    # Parse response - handle markdown code blocks
    response_text = image_response.text.strip()

    if response_text.startswith("```"):
        # Remove markdown code blocks
        response_text = response_text.split("```")[1]
        if response_text.startswith("json"):
            response_text = response_text[4:].strip()

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON array from text
        import re
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            return []


def visualize_annotations(
    image, annotations: list, output_path: str = None, roi_box_size: int = 60
):
    """
    Draw ROI boxes on image for detected objects.

    Args:
        image: Either a PIL Image object or a file path string
        annotations: List of annotations in format [{"point": [y, x], "label": "..."}, ...]
        output_path: Optional path to save the annotated image (if None, returns Image object)
        roi_box_size: Size of the box around the center point (default: 60)

    Returns:
        PIL Image object with annotations drawn (if output_path is None)
        None (if output_path is provided, saves to file instead)
    """
    # Handle both Image objects and file paths
    if isinstance(image, str):
        try:
            img = Image.open(image)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image}")
            return None
    else:
        # It's already an Image object
        img = image

    draw = ImageDraw.Draw(img)
    width, height = img.size

    roi_color = (0, 255, 100)  # Greenish color

    for annotation in annotations:
        y_norm, x_norm = annotation["point"]
        label = annotation.get("label", "object")

        x_pixel = int(x_norm * width / 1000)
        y_pixel = int(y_norm * height / 1000)

        # Calculate ROI box coordinates
        x1 = max(0, x_pixel - roi_box_size)
        y1 = max(0, y_pixel - roi_box_size)
        x2 = min(width, x_pixel + roi_box_size)
        y2 = min(height, y_pixel + roi_box_size)

        # Draw greenish ROI box
        draw.rectangle([x1, y1, x2, y2], outline=roi_color, width=4)

    # Save to file if output_path is provided, otherwise return the image
    if output_path:
        img.save(output_path)
        print(f"Annotated image saved successfully as '{output_path}'")
        return None
    else:
        return img


if __name__ == "__main__":
    MODEL_ID = "gemini-robotics-er-1.5-preview"
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    with open("imgs/img.png", "rb") as f:
        image_bytes = f.read()

    queries = ["safety helmet", "hard hat"]

    # Run inference
    annotations = infer(MODEL_ID, client, image_bytes, queries, mime_type="image/png")
    print(f"Found {len(annotations)} objects")

    # Visualize and save
    visualize_annotations("imgs/img.png", annotations, output_path="annotated_output.png")
