from PIL import Image, ImageDraw, ImageFont
import numpy as np
import copy




def prediction_to_box_coordinates(prediction, image_width, image_height):
    """
    Convert a prediction dictionary with center coordinates to corner coordinates

    Args:
        prediction: Dictionary containing x, y, width, height in pixel values
        image_width: Width of the target image
        image_height: Height of the target image

    Returns:
        Tuple of (x1, y1, x2, y2) coordinates for the box corners
    """
    # Get coordinates
    x_center = prediction.get("x", 0)
    y_center = prediction.get("y", 0)
    w = prediction.get("width", 0)
    h = prediction.get("height", 0)

    # Calculate box coordinates, constrained to image boundaries
    x1 = int(max(0, x_center - w/2))
    y1 = int(max(0, y_center - h/2))
    x2 = int(min(image_width, x_center + w/2))
    y2 = int(min(image_height, y_center + h/2))

    return x1, y1, x2, y2

def draw_roboflow_predictions(image, predictions_list):
    """
    Draw Roboflow API predictions on a PIL Image using Pillow

    Args:
        image: PIL Image object (must be in RGB mode or will be converted)
        predictions_list: List of prediction dictionaries from Roboflow API

    Returns:
        PIL Image with predictions drawn on it
    """
    # Create a copy of the image and convert to RGB mode for colored annotations
    if image.mode != 'RGB':
        image_copy = image.convert('RGB')
    else:
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

    # Draw each prediction
    for pred in predictions_list:
        # Get box coordinates using the new helper function
        x1, y1, x2, y2 = prediction_to_box_coordinates(pred, width, height)

        # Get confidence and class information
        confidence = pred.get("confidence", 0)
        class_name = pred.get("class", "unknown")
        class_id = pred.get("class_id", 0)

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

def scale_predictions_to_resolution(predictions_list, target_resolution):
    """
    Scale predictions from 1024x1024 model input to a higher resolution square image

    Args:
        predictions_list: List of prediction dictionaries from Roboflow API
        target_resolution: Int or tuple of target resolution (assumes square if int)

    Returns:
        New list of predictions with scaled coordinates
    """
    # Handle both int and tuple input for target_resolution
    if isinstance(target_resolution, int):
        target_width = target_height = target_resolution
    else:
        target_width, target_height = target_resolution

    # Calculate scale factors (from 1024x1024)
    scale_x = target_width / 1024
    scale_y = target_height / 1024

    # Create a deep copy of predictions to avoid modifying the original
    scaled_predictions = copy.deepcopy(predictions_list)

    # Scale each prediction
    for pred in scaled_predictions:
        # Scale coordinates
        pred["x"] = pred.get("x", 0) * scale_x
        pred["y"] = pred.get("y", 0) * scale_y
        pred["width"] = pred.get("width", 0) * scale_x
        pred["height"] = pred.get("height", 0) * scale_y

    return scaled_predictions

# Example usage of the new function:
"""
# Get a single prediction
prediction = {
    "x": 478.0,
    "y": 475.5,
    "width": 436.0,
    "height": 107.0,
    "confidence": 0.95,
    "class": "mcq"
}

# Get the box coordinates
img_width, img_height = 1024, 1024
x1, y1, x2, y2 = prediction_to_box_coordinates(prediction, img_width, img_height)
print(f"Box coordinates: ({x1}, {y1}) to ({x2}, {y2})")
"""
