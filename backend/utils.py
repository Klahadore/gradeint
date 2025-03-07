from PIL import Image, ImageDraw, ImageFont
import numpy as np
import copy
import os
from pypdf import PdfReader, PdfWriter




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




def extract_first_pages(pdf_path, num_pages, output_dir, output_filename=None):
    """
    Extract the first specified number of pages from a PDF, save them to a new file,
    and remove those pages from the original PDF.

    Args:
        pdf_path: Path to the original PDF file
        num_pages: Number of pages to extract from the beginning
        output_dir: Directory where the extracted pages will be saved
        output_filename: Optional name for the output file (defaults to the original filename with "_extract" suffix)

    Returns:
        Tuple containing paths to the extracted PDF and the modified original PDF
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine output filename if not provided
    if output_filename is None:
        base_name = os.path.basename(pdf_path)
        name_without_ext, ext = os.path.splitext(base_name)
        output_filename = f"{name_without_ext}_extract{ext}"

    output_path = os.path.join(output_dir, output_filename)

    # Read the original PDF
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        total_pages = len(pdf_reader.pages)

        # Check if there are enough pages
        if num_pages > total_pages:
            raise ValueError(f"PDF only has {total_pages} pages, but {num_pages} were requested")

        # Create a new PDF with the first N pages
        extracted_writer = PdfWriter()
        for page_num in range(num_pages):
            extracted_writer.add_page(pdf_reader.pages[page_num])

        # Create another PDF with the remaining pages
        remaining_writer = PdfWriter()
        for page_num in range(num_pages, total_pages):
            remaining_writer.add_page(pdf_reader.pages[page_num])

        # Save the extracted pages to the output directory
        with open(output_path, 'wb') as output_file:
            extracted_writer.write(output_file)

        # Create a temporary file for the remaining pages
        temp_path = pdf_path + ".temp"
        with open(temp_path, 'wb') as temp_file:
            remaining_writer.write(temp_file)

    # Replace the original file with the remaining pages
    os.replace(temp_path, pdf_path)

    return output_path, pdf_path

if __name__ == "__main__":
    pdf_file = "temp_files/test_util/frq_8.pdf"
    output_directory = "temp_files/test_util/"


    extracted_file, modified_original = extract_first_pages(
        pdf_file,
        3,
        output_directory,
        "first_three_pages.pdf"
    )
