from PIL import Image, ImageDraw, ImageFont
import numpy as np

def draw_roboflow_predictions(image, predictions_list):
    """
    Draw Roboflow API predictions on a PIL Image using Pillow

    Args:
        image: PIL Image object
        predictions_list: List of prediction dictionaries from Roboflow API

    Returns:
        PIL Image with predictions drawn on it
    """
    # Create a copy of the image to draw on
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    # Get image dimensions
    width, height = image_copy.size

    # Try to load a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Generate colors for classes (create enough colors for all potential classes)
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)  # Assume max 100 classes

    # Keep track of classes we've seen
    class_mapping = {}

    # Draw each prediction
    for pred in predictions_list:
        # Extract information based on common Roboflow formats
        if "bbox" in pred:
            # Format with bbox object
            bbox = pred["bbox"]
            x_center = bbox.get("x", 0)
            y_center = bbox.get("y", 0)
            w = bbox.get("width", 0)
            h = bbox.get("height", 0)
        else:
            # Direct format
            x_center = pred.get("x", 0)
            y_center = pred.get("y", 0)
            w = pred.get("width", 0)
            h = pred.get("height", 0)

        # Get confidence and class information
        confidence = pred.get("confidence", 0)
        class_name = pred.get("class", "unknown")

        # Assign a unique ID to each class name if not already assigned
        if class_name not in class_mapping:
            class_mapping[class_name] = len(class_mapping)
        class_id = class_mapping[class_name]

        # Check if coordinates are normalized (0-1) or in pixels
        # If all values are <= 1, assume normalized
        if all(val <= 1 for val in [x_center, y_center, w, h]):
            # Convert normalized coordinates to pixel values
            x_center *= width
            y_center *= height
            w *= width
            h *= height

        # Calculate box coordinates
        x1 = int(max(0, x_center - w/2))
        y1 = int(max(0, y_center - h/2))
        x2 = int(min(width, x_center + w/2))
        y2 = int(min(height, y_center + h/2))

        # Get color for this class
        color = tuple(map(int, colors[class_id % len(colors)]))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Create label
        label = f"{class_name}: {confidence:.2f}"

        # Get text size
        text_size = draw.textbbox((0, 0), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]

        # Draw label background
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width, y1],
            fill=color
        )

        # Draw label text
        draw.text((x1, y1 - text_height - 2), label, fill=(255, 255, 255), font=font)

    return image_copy
