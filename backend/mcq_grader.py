import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
import asyncio
from utils import *
from yolo_inference import *
import pathlib
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, ClassVar
import copy
import json
from yolo_inference import process_raw_png, square_original_image
# Load API key from environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {bool(google_api_key)}")

# Initialize Google Gemini client
client = genai.Client(api_key=google_api_key)




class WorksheetPage(BaseModel):
    """Represents a single page in a worksheet with both original and processed versions"""
    worksheet_index: int
    page_index: int
    original_path: str = None
    processed_path: str = None
    _original_image: Optional[Image.Image] = None
    _processed_image: Optional[Image.Image] = None

    class Config:
        arbitrary_types_allowed = True

    def get_original_image(self) -> Image.Image:
        """Loads and returns the original image if not already loaded or was closed"""
        # Always reopen the image if it's closed or None
        if self._original_image is None and self.original_path:
            self._original_image = Image.open(self.original_path)
        # Check if the image is closed and reopen if needed
        elif hasattr(self._original_image, 'im') and self._original_image.im is None and self.original_path:
            self._original_image = Image.open(self.original_path)
        return self._original_image

    def get_processed_image(self) -> Image.Image:
        """Loads and returns the processed image if not already loaded"""
        if self._processed_image is None and self.processed_path:
            self._processed_image = Image.open(self.processed_path)
        return self._processed_image

    def set_original_image(self, image: Image.Image, path: str):
        """Sets the original image and its path"""
        self._original_image = image
        self.original_path = path

    def set_processed_image(self, image: Image.Image, path: str):
        """Sets the processed image and its path"""
        self._processed_image = image
        self.processed_path = path

    def close_images(self):
        pass
        # """Closes both images to free memory"""
        # if self._original_image:
        #     self._original_image.close()
        #     self._original_image = None
        # if self._processed_image:
        #     self._processed_image.close()
        #     self._processed_image = None

class WorksheetCollection(BaseModel):
    """Collection of all worksheet pages, organized for easy access"""
    pages: Dict[str, WorksheetPage] = Field(default_factory=dict)

    def get_or_create_page(self, worksheet_idx: int, page_idx: int) -> WorksheetPage:
        """Gets an existing page or creates a new one if it doesn't exist"""
        key = f"{worksheet_idx}_{page_idx}"
        if key not in self.pages:
            self.pages[key] = WorksheetPage(
                worksheet_index=worksheet_idx,
                page_index=page_idx
            )
        return self.pages[key]

    def get_worksheets(self) -> List[List[WorksheetPage]]:
        """Returns a list of worksheets, each containing a list of pages"""
        # Group pages by worksheet_index
        worksheets = {}
        for key, page in self.pages.items():
            if page.worksheet_index not in worksheets:
                worksheets[page.worksheet_index] = []
            worksheets[page.worksheet_index].append(page)

        # Sort worksheets by index and sort pages within each worksheet
        result = []
        for idx in sorted(worksheets.keys()):
            pages = sorted(worksheets[idx], key=lambda p: p.page_index)
            result.append(pages)

        return result

    def close_all_images(self):
        pass
        # """Closes all loaded images to free memory"""
        # for page in self.pages.values():
        #     page.close_images()


def process_worksheets_to_png(whole_worksheets: str, output_dir: str, worksheet_len: int,
                              dpi=300, collection: Optional[WorksheetCollection] = None) -> WorksheetCollection:
    """
    Split a large PDF into individual worksheets with unprocessed images.

    Args:
        whole_worksheets: Path to the combined PDF file containing all worksheets
        output_dir: Directory where worksheet folders will be created
        worksheet_len: Number of pages in each worksheet
        dpi: Resolution for the PNG images (default: 300)
        collection: Optional existing WorksheetCollection to add to

    Returns:
        WorksheetCollection containing all the pages
    """
    # Create a new collection if none was provided
    if collection is None:
        collection = WorksheetCollection()

    # Ensure output directory exists
    output_base = pathlib.Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Open the combined PDF
    pdf_reader = PdfReader(whole_worksheets)
    total_pages = len(pdf_reader.pages)

    # Calculate number of worksheets
    num_worksheets = (total_pages + worksheet_len - 1) // worksheet_len  # Ceiling division

    for worksheet_idx in range(num_worksheets):
        # Create a folder for this worksheet
        worksheet_folder = output_base / f"worksheet_{worksheet_idx + 1}"
        worksheet_folder.mkdir(exist_ok=True)

        # Calculate page range for this worksheet
        start_page = worksheet_idx * worksheet_len
        end_page = min(start_page + worksheet_len, total_pages)

        # Create a temporary PDF for this worksheet
        temp_pdf_path = worksheet_folder / "temp_worksheet.pdf"
        pdf_writer = PdfWriter()

        # Add the worksheet's pages
        for page_num in range(start_page, end_page):
            pdf_writer.add_page(pdf_reader.pages[page_num])

        # Save the temporary PDF
        with open(temp_pdf_path, 'wb') as temp_file:
            pdf_writer.write(temp_file)

        # Convert PDF pages to PNG
        try:
            # Convert PDF to images
            images = convert_from_path(str(temp_pdf_path), dpi=dpi)

            # Save each page as PNG
            for i, image in enumerate(images):
                page_num = i + 1
                page_idx = i

                # Get or create the page in our collection
                page = collection.get_or_create_page(worksheet_idx, page_idx)

                # Process and save the image
                png_path = worksheet_folder / f"page_{page_num}.png"
                square_image = square_original_image(image)
                square_image.save(str(png_path), "PNG")

                # Update the page in our collection
                page.set_original_image(square_image, str(png_path))

            # Delete the temporary PDF
            temp_pdf_path.unlink()

        except Exception as e:
            print(f"Error converting worksheet {worksheet_idx + 1} to PNGs: {str(e)}")
            # Keep the PDF if conversion fails
            pass

    return collection



async def scan_single_slice(client, img, idx):
    # Run the API call in a separate thread to avoid blocking
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,  # Uses default ThreadPoolExecutor
        lambda: client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Which answer choice is circled? Answer with the capital answer choice letter. If there are multiple answers selected, return 'null'", img]
        )
    )

    extracted_text = response.text.strip()
  #  print(f"Question {idx+1}: {extracted_text}")
    return extracted_text

async def scan_images(images: list[Image.Image], client):
    # Create tasks for all images
    tasks = [
        scan_single_slice(client, img, idx)
        for idx, img in enumerate(images)
    ]

    # Run all tasks concurrently and wait for all to complete
    model_responses = await asyncio.gather(*tasks)

    # Final output

    return model_responses


def clean_outputs(outputs: list[str]) -> list[str]:
    for i in range(len(outputs)):
        outputs[i] = outputs[i].upper()
        if outputs[i] not in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            outputs[i] = "null"

    return outputs

async def main():
    img = Image.open("dataset/third_data_pngs/ap-english-language-and-composition-course-description - AP Lang Multi_page_16.png")
    processed_img = process_raw_png(img)
    square_image = square_original_image(img)

    predictions = inference_on_img(processed_img, square_image.size)
    print(predictions)
    # square_image = draw_roboflow_predictions(square_image, predictions)

    images = extract_image_slices(square_image, predictions)
    # Now we can use await inside this async function
    model_responses = await scan_images(images, client)

    # Continue with any processing using model_responses
    print(model_responses)
    return model_responses

# returns coordinates from answer key obtained from
def process_answer_key(rubric_path: str):
    """
    Process a PDF rubric to extract answer choices and their coordinates.

    Args:
        rubric_path: Path to the PDF rubric file

    Returns:
        tuple: (answer_choices, prediction_coords)
            - answer_choices: 2D array of cleaned answer choices
            - prediction_coords: 2D array of prediction coordinates
    """
    # Convert PDF to images
    images = convert_from_path(
        rubric_path,
        output_folder=None,
        fmt="ppm",
        use_pdftocairo=True,
        transparent=False,
        poppler_path=None,
        dpi=600
    )

    # Initialize result containers
    prediction_coords = []
    answer_choices = []

    # Process each page of the PDF
    for i, image in enumerate(images):
        # Save original image size and create squared version
        orig_image_size = image.size
        square_image = square_original_image(image)

        # Process image for inference
        processed_image = process_raw_png(image)

        # Get predictions (bounding boxes)
        predictions = inference_on_img(processed_image, orig_image_size)
        prediction_coords.append(predictions)

        # Extract image slices from the predictions
        slices = extract_image_slices(square_image, predictions)

        # Run inference on all slices for this page
        # Using asyncio.run to call our async function from sync context
        page_answers = asyncio.run(scan_images(slices, client))

        # Clean the outputs to standardize answer format
        cleaned_answers = clean_outputs(page_answers)
        answer_choices.append(cleaned_answers)

        # Free up memory - DISABLED
        # image.close()
        # square_image.close()
        # processed_image.close()

    for i in range(len(answer_choices)):
        answer_choices[i] = clean_outputs(answer_choices[i])
    return answer_choices, prediction_coords


def grade_student_worksheets(
    worksheet_collection: WorksheetCollection,
    answer_choices: List[List[str]],
    prediction_coords: List[List[dict]]
) -> tuple[List[List[List[str]]], List[List[List[dict]]]]:
    """
    Process student worksheets using coordinates from the answer key.

    Args:
        worksheet_collection: Collection of worksheets separated by student
        answer_choices: 2D array of answer choices from the answer key
        prediction_coords: 2D array of bounding box coordinates from the answer key

    Returns:
        Tuple containing:
        - List[List[List[str]]]: 3D array of student answers (student -> page -> answers)
        - List[List[List[dict]]]: 3D array of prediction coordinates (student -> page -> coords)
    """
    all_student_answers = []
    all_student_coords = []

    # Get worksheets organized by student
    worksheets = worksheet_collection.get_worksheets()

    print(f"Processing {len(worksheets)} student worksheets")

    # Process each student's worksheet
    for worksheet_idx, worksheet_pages in enumerate(worksheets):
        student_answers_by_page = []
        student_coords_by_page = []

        print(f"Student {worksheet_idx+1} has {len(worksheet_pages)} pages")

        # Process each page of this student's worksheet
        for page_idx, page in enumerate(worksheet_pages):
            # Get the original image
            original_image = page.get_original_image()
            if original_image is None:
                print(f"Warning: Missing image for worksheet {worksheet_idx+1}, page {page_idx+1}")
                continue

            # Map to the corresponding answer key page (in case answer key has fewer pages)
            key_page_idx = page_idx % len(prediction_coords)

            print(f"Processing student {worksheet_idx+1}, page {page_idx+1} (mapping to key page {key_page_idx+1})")

            # Get coordinates from the answer key for this page
            page_coords = prediction_coords[key_page_idx]

            # Store the coordinates for this page (maintain the exact structure)
            student_coords_by_page.append(page_coords)

            # Extract the relevant slices from the student's page
            answer_slices = extract_image_slices(original_image, page_coords)

            if not answer_slices:
                print(f"Warning: No answer slices extracted for student {worksheet_idx+1}, page {page_idx+1}")
                student_answers_by_page.append([])
                continue

            print(f"Extracted {len(answer_slices)} answer slices for student {worksheet_idx+1}, page {page_idx+1}")

            # Use scan_images to get the student's answers for this page
            try:
                page_responses = asyncio.run(scan_images(answer_slices, client))
                # Clean the responses
                cleaned_responses = clean_outputs(page_responses)
                # Add this page's answers to the student's by-page answers
                student_answers_by_page.append(cleaned_responses)
                print(f"Successfully processed {len(cleaned_responses)} answers for student {worksheet_idx+1}, page {page_idx+1}")
            except Exception as e:
                print(f"Error processing answers for student {worksheet_idx+1}, page {page_idx+1}: {str(e)}")
                student_answers_by_page.append([])

            # Free up memory - DISABLED
            # page.close_images()

        # Add this student's answers to the overall results
        all_student_answers.append(student_answers_by_page)
        all_student_coords.append(student_coords_by_page)

    print(f"Finished processing {len(all_student_answers)} students")
    print(f"Structure check - Student answers: {len(all_student_answers)} students")
    for i, student in enumerate(all_student_answers):
        print(f"  Student {i+1}: {len(student)} pages")
        for j, page in enumerate(student):
            print(f"    Page {j+1}: {len(page)} answers")

    return all_student_answers, all_student_coords

def process_and_grade_worksheets(
    rubric_path: str,
    worksheet_pdf_path: str,
    output_dir: str,
    pages_per_student: int
):
    """
    Process a rubric and grade student worksheets.

    Args:
        rubric_path: Path to the answer key PDF
        worksheet_pdf_path: Path to the PDF containing all student worksheets
        output_dir: Directory to store processed images
        pages_per_student: Number of pages per student worksheet

    Returns:
        Tuple containing:
        - List[List[List[str]]]: 3D array of student answers (student -> page -> answers)
        - List[List[List[dict]]]: 3D array of prediction coordinates (student -> page -> coords)
    """
    # Process the answer key
    print("Processing answer key...")
    answer_choices, prediction_coords = process_answer_key(rubric_path)
    print(f"Answer key has {len(answer_choices)} pages with answers")

    for i, page_coords in enumerate(prediction_coords):
        print(f"  Page {i+1} has {len(page_coords)} answer coordinates")

    # Process the student worksheets
    print(f"Processing student worksheets from {worksheet_pdf_path}...")
    collection = WorksheetCollection()
    process_worksheets_to_png(
        worksheet_pdf_path,
        output_dir,
        pages_per_student,
        dpi=600,
        collection=collection
    )

    worksheets = collection.get_worksheets()
    print(f"Split into {len(worksheets)} student worksheets")

    # Grade the worksheets
    print("Grading student worksheets...")
    student_answers, student_coords = grade_student_worksheets(
        collection,
        answer_choices,
        prediction_coords
    )

    # Clean up - DISABLED
    # collection.close_all_images()

    return student_answers, student_coords

import os
from PIL import Image, ImageDraw
import pathlib


def create_graded_worksheets(
    worksheet_collection: WorksheetCollection,
    student_answers: List[List[List[str]]],
    answer_key: List[List[str]],
    student_coords: List[List[List[dict]]],
    output_dir: str,
    assets_dir: str = "checkandx"  # Directory containing checkmark and X images
) -> List[str]:
    """
    Create graded worksheets with checkmarks for correct answers and X's for incorrect ones.
    """
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and resize check and X assets
    check_path = os.path.join(assets_dir, "check.png")
    x_path = os.path.join(assets_dir, "x.png")

    try:
        check_img = Image.open(check_path).convert("RGBA")
        x_img = Image.open(x_path).convert("RGBA")

        # Resize to 64x64
        check_img = check_img.resize((64, 64), Image.Resampling.LANCZOS)
        x_img = x_img.resize((64, 64), Image.Resampling.LANCZOS)

        print(f"Loaded check and X assets from {assets_dir}")
    except Exception as e:
        print(f"Error loading assets: {str(e)}")
        print("Falling back to drawing shapes directly")
        check_img = None
        x_img = None

    # Get worksheets
    worksheets = worksheet_collection.get_worksheets()

    # Track all output files in order
    output_files = []

    # Process each student
    for student_idx, student_pages in enumerate(worksheets):
        # Skip if we don't have answers for this student
        if student_idx >= len(student_answers):
            continue

        # Get this student's answers
        student_page_answers = student_answers[student_idx]
        student_page_coords = student_coords[student_idx]

        # Process each page
        for page_idx, page in enumerate(student_pages):
            # Skip if we don't have answers for this page
            if page_idx >= len(student_page_answers):
                continue

            # Get original image to determine size
            original_image = page.get_original_image()
            if original_image is None:
                continue

            print(f"\nProcessing Student {student_idx+1}, Page {page_idx+1}")
            print(f"Image dimensions: {original_image.width} x {original_image.height}")

            # Get key page index (in case answer key has fewer pages)
            key_page_idx = page_idx % len(answer_key)

            # Create a transparent image same size as original
            graded_image = Image.new('RGBA', original_image.size, (0, 0, 0, 0))

            # Get page answers and coordinates
            page_answers = student_page_answers[page_idx]
            page_coords = student_page_coords[page_idx]
            key_answers = answer_key[key_page_idx]

            print(f"Found {len(page_coords)} answer coordinates and {len(page_answers)} answers")

            # Draw check marks or X's for each answer
            for answer_idx, (student_answer, key_answer) in enumerate(zip(page_answers, key_answers)):
                # Skip if we're out of bounds
                if answer_idx >= len(page_coords):
                    continue

                # Get the bounding box
                bbox = page_coords[answer_idx]

                # Print the raw prediction data
                print(f"\nAnswer {answer_idx+1}:")
                print(f"  Raw prediction: {bbox}")

                # Convert center coordinates to corner coordinates
                x1, y1, x2, y2 = prediction_to_box_coordinates(
                    bbox,
                    original_image.width,
                    original_image.height
                )

                print(f"  Box corners: ({x1}, {y1}) to ({x2}, {y2})")

                # Calculate a better position for the mark:
                # Let's place it in the top-left corner of the answer box, with a small offset
                mark_size = 128  # Size of our check/X images
                offset = 5  # Small offset to not completely overlap the answer circle

                mark_x = x1 + offset
                mark_y = y1 + offset

                # Alternatively, place it in the center of the answer box
                # mark_x = (x1 + x2) // 2 - mark_size // 2
                # mark_y = (y1 + y2) // 2 - mark_size // 2

                print(f"  Placing mark at: ({mark_x}, {mark_y})")
                print(f"  Student answer: {student_answer}, Key answer: {key_answer}")

                # Determine if the answer is correct
                is_correct = (student_answer == key_answer) and (student_answer != "null")
                print(f"  Is correct: {is_correct}")

                # Adjust mark size based on the box size
                box_width = x2 - x1
                box_height = y2 - y1
                adaptive_mark_size = min(mark_size, box_width // 2, box_height // 2)
                if adaptive_mark_size < 20:  # Ensure minimum visibility
                    adaptive_mark_size = 20

                # Use image assets if available, otherwise draw shapes
                if is_correct and check_img is not None:
                    # Resize the checkmark based on the box size
                    resized_check = check_img.resize((adaptive_mark_size, adaptive_mark_size),
                                                     Image.Resampling.LANCZOS)
                    # Paste checkmark at the calculated position
                    graded_image.paste(resized_check, (mark_x, mark_y), resized_check)
                elif not is_correct and x_img is not None:
                    # Resize the X based on the box size
                    resized_x = x_img.resize((adaptive_mark_size, adaptive_mark_size),
                                             Image.Resampling.LANCZOS)
                    # Paste X at the calculated position
                    graded_image.paste(resized_x, (mark_x, mark_y), resized_x)
                else:
                    # Fallback to drawing shapes if assets can't be loaded
                    draw = ImageDraw.Draw(graded_image)

                    if is_correct:
                        # Draw green checkmark
                        color = (0, 200, 0, 255)  # Green with full opacity
                        line_width = max(3, adaptive_mark_size // 8)
                        draw.line(
                            [
                                (mark_x, mark_y + adaptive_mark_size/2),
                                (mark_x + adaptive_mark_size/3, mark_y + adaptive_mark_size),
                                (mark_x + adaptive_mark_size, mark_y)
                            ],
                            fill=color,
                            width=line_width
                        )
                    else:
                        # Draw red X
                        color = (255, 0, 0, 255)  # Red with full opacity
                        line_width = max(3, adaptive_mark_size // 8)
                        draw.line(
                            [(mark_x, mark_y), (mark_x + adaptive_mark_size, mark_y + adaptive_mark_size)],
                            fill=color,
                            width=line_width
                        )
                        draw.line(
                            [(mark_x, mark_y + adaptive_mark_size), (mark_x + adaptive_mark_size, mark_y)],
                            fill=color,
                            width=line_width
                        )

            # Save the overlay image
            output_filename = f"student_{student_idx+1:03d}_page_{page_idx+1:03d}.png"
            output_file_path = output_path / output_filename
            graded_image.save(str(output_file_path), "PNG")
            output_files.append(str(output_file_path))
            print(f"Saved overlay to {output_file_path}")

            # Composite with original image to create a preview
            preview_image = Image.alpha_composite(
                original_image.convert("RGBA"),
                graded_image
            )
            preview_filename = f"preview_student_{student_idx+1:03d}_page_{page_idx+1:03d}.png"
            preview_file_path = output_path / preview_filename
            preview_image.save(str(preview_file_path), "PNG")
            print(f"Saved preview to {preview_file_path}")

    return output_files


def create_flat_worksheet_folder(
    worksheet_collection: WorksheetCollection,
    output_dir: str
) -> List[str]:
    """
    Create a flat folder containing all worksheet images with consistent naming.

    Args:
        worksheet_collection: Collection of student worksheets
        output_dir: Directory to save the flat worksheet structure

    Returns:
        List of paths to copied images in order
    """
    # Create output directory
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get worksheets
    worksheets = worksheet_collection.get_worksheets()

    # Track all output files in order
    output_files = []

    # Process each student
    for student_idx, student_pages in enumerate(worksheets):
        # Process each page
        for page_idx, page in enumerate(student_pages):
            # Get original image
            original_image = page.get_original_image()
            if original_image is None:
                continue

            # Create new filename
            output_filename = f"student_{student_idx+1:03d}_page_{page_idx+1:03d}.png"
            output_file_path = output_path / output_filename

            # Save the image
            original_image.save(str(output_file_path), "PNG")
            output_files.append(str(output_file_path))

            # Free up memory - DISABLED
            # original_image.close()

    return output_files

import os
import re
import pathlib
from PIL import Image
from typing import List, Tuple, Optional
import img2pdf

def natural_sort_key(s):
    """
    Sort strings that contain numbers in a human-friendly way.
    For example: ["page_1.png", "page_10.png", "page_2.png"] -> ["page_1.png", "page_2.png", "page_10.png"]
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def combine_images_to_pdf(
    image_files: List[str],
    output_pdf_path: str,
    target_aspect_ratio: float = 0.77,  # US Letter aspect ratio (8.5/11)
    dpi: int = 300,
    crop_mode: str = "center"  # "center", "top", or "bottom"
) -> str:
    """
    Combines square PNG images into a single PDF with proper document aspect ratio.

    Args:
        image_files: List of paths to PNG images
        output_pdf_path: Path where the output PDF will be saved
        target_aspect_ratio: Desired width/height ratio (default: US Letter 0.77)
        dpi: Resolution for the PDF (dots per inch)
        crop_mode: How to crop the square images ("center", "top", or "bottom")

    Returns:
        Path to the created PDF file
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_pdf_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Sort files naturally
    sorted_files = sorted(image_files, key=natural_sort_key)

    if not sorted_files:
        raise ValueError("No image files provided")

    print(f"Processing {len(sorted_files)} images...")

    # Create a list to store processed images
    processed_images = []
    temp_files = []

    try:
        # Process each image
        for i, img_path in enumerate(sorted_files):
            print(f"Processing image {i+1}/{len(sorted_files)}: {img_path}")

            # Open image
            with Image.open(img_path) as img:
                # Check if image is square-ish (allow small differences)
                width, height = img.size

                # If it's already very close to our target aspect ratio, don't crop
                current_aspect = width / height
                if abs(current_aspect - target_aspect_ratio) < 0.05:
                    print(f"  Image already has suitable aspect ratio ({current_aspect:.2f}), skipping crop")
                    processed_img = img.copy()
                else:
                    # Calculate new dimensions
                    if current_aspect > target_aspect_ratio:
                        # Image is too wide, crop width
                        new_width = int(height * target_aspect_ratio)
                        new_height = height
                    else:
                        # Image is too tall, crop height
                        new_width = width
                        new_height = int(width / target_aspect_ratio)

                    # Calculate crop box
                    if crop_mode == "center":
                        # Crop equally from both sides/top and bottom
                        left = (width - new_width) // 2
                        top = (height - new_height) // 2
                    elif crop_mode == "top":
                        # Crop from sides and keep top
                        left = (width - new_width) // 2
                        top = 0
                    elif crop_mode == "bottom":
                        # Crop from sides and keep bottom
                        left = (width - new_width) // 2
                        top = height - new_height
                    else:
                        raise ValueError(f"Invalid crop_mode: {crop_mode}")

                    right = left + new_width
                    bottom = top + new_height

                    # Crop the image
                    processed_img = img.crop((left, top, right, bottom))
                    print(f"  Cropped from {width}x{height} to {new_width}x{new_height}")

                # Convert to RGB if needed for PDF compatibility
                if processed_img.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', processed_img.size, (255, 255, 255))
                    # Paste the image on the background
                    background.paste(processed_img, mask=processed_img.split()[3])
                    processed_img = background
                elif processed_img.mode != 'RGB':
                    processed_img = processed_img.convert('RGB')

                # Save to temporary file
                temp_filename = f"temp_proc_{i}.jpg"
                temp_filepath = os.path.join(output_dir, temp_filename)
                processed_img.save(temp_filepath, "JPEG", quality=95, dpi=(dpi, dpi))
                temp_files.append(temp_filepath)
                processed_images.append(temp_filepath)

        # Create PDF from processed images using img2pdf for better quality
        print(f"Creating PDF with {len(processed_images)} images...")

        with open(output_pdf_path, "wb") as f:
            # Use img2pdf with specified DPI
            layout_fun = img2pdf.get_layout_fun(lambda pts: (pts[0], pts[1] * (1/target_aspect_ratio)))
            f.write(img2pdf.convert(processed_images, layout_fun=layout_fun, dpi=dpi))

        print(f"PDF created successfully: {output_pdf_path}")
        return output_pdf_path

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_file}: {e}")

# Alternative version using reportlab if img2pdf is not available
def combine_images_to_pdf_reportlab(
    image_files: List[str],
    output_pdf_path: str,
    target_aspect_ratio: float = 0.77,
    dpi: int = 300,
    crop_mode: str = "center"
) -> str:
    """
    Combines square PNG images into a single PDF using reportlab.
    """
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    # Sort files naturally
    sorted_files = sorted(image_files, key=natural_sort_key)

    if not sorted_files:
        raise ValueError("No image files provided")

    # Get dimensions from first image
    with Image.open(sorted_files[0]) as first_img:
        width, height = first_img.size

    # Calculate PDF page size (8.5 x 11 inches by default)
    page_width = 8.5 * inch
    page_height = page_width / target_aspect_ratio

    # Create PDF
    c = canvas.Canvas(output_pdf_path, pagesize=(page_width, page_height))

    for img_path in sorted_files:
        with Image.open(img_path) as img:
            # Perform cropping as in the original function
            width, height = img.size

            # Calculate new dimensions based on target aspect ratio
            if width / height > target_aspect_ratio:
                new_width = int(height * target_aspect_ratio)
                new_height = height
            else:
                new_width = width
                new_height = int(width / target_aspect_ratio)

            # Calculate crop box
            if crop_mode == "center":
                left = (width - new_width) // 2
                top = (height - new_height) // 2
            elif crop_mode == "top":
                left = (width - new_width) // 2
                top = 0
            elif crop_mode == "bottom":
                left = (width - new_width) // 2
                top = height - new_height
            else:
                raise ValueError(f"Invalid crop_mode: {crop_mode}")

            right = left + new_width
            bottom = top + new_height

            # Crop the image
            processed_img = img.crop((left, top, right, bottom))

            # Save to temporary file
            temp_filename = f"temp_{os.path.basename(img_path)}"
            processed_img.save(temp_filename)

            # Draw on canvas
            c.drawImage(temp_filename, 0, 0, page_width, page_height)
            c.showPage()

            # Delete temporary file
            os.remove(temp_filename)

    c.save()
    return output_pdf_path


def process_grade_and_create_pdfs(
    rubric_path: str,
    student_worksheets_path: str,
    output_dir: str,
    pages_per_student: int,
    assets_dir: str = "checkandx",
    dpi: int = 600,
    target_aspect_ratio: float = 0.77
) -> dict:
    """
    Process answer key, grade student worksheets, and create PDFs with results.

    Args:
        rubric_path: Path to the answer key PDF
        student_worksheets_path: Path to the PDF containing all student worksheets
        output_dir: Directory to store processed images and PDFs
        pages_per_student: Number of pages per student worksheet
        assets_dir: Directory containing check.png and x.png assets
        dpi: Resolution for image processing (default: 600)
        target_aspect_ratio: Aspect ratio for PDF creation (default: 0.77 for US Letter)

    Returns:
        Dictionary containing paths to created files and results:
            - 'graded_pdf': Path to the PDF with graded worksheets
            - 'overlay_pdf': Path to the PDF with just the overlay marks
            - 'worksheets_dir': Directory containing processed worksheet images
            - 'graded_dir': Directory containing graded overlay images
            - 'flat_dir': Directory containing flat worksheet images
            - 'student_answers': 3D array of student answers
            - 'answer_key': 2D array of answer key choices
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    worksheets_dir = os.path.join(output_dir, "worksheets")
    graded_dir = os.path.join(output_dir, "graded_overlays")
    flat_dir = os.path.join(output_dir, "flat_worksheets")

    # Result paths
    graded_pdf_path = os.path.join(output_dir, "graded_worksheets.pdf")
    overlay_pdf_path = os.path.join(output_dir, "overlay_only.pdf")

    # Process the answer key
    print("Processing answer key...")
    answer_choices, prediction_coords = process_answer_key(rubric_path)
    print(f"Answer key has {len(answer_choices)} pages with answers")

    for i, page_coords in enumerate(prediction_coords):
        print(f"  Page {i+1} has {len(page_coords)} answer coordinates")

    # Process the student worksheets
    print(f"Processing student worksheets from {student_worksheets_path}...")
    collection = WorksheetCollection()
    process_worksheets_to_png(
        student_worksheets_path,
        worksheets_dir,
        pages_per_student,
        dpi=dpi,
        collection=collection
    )

    worksheets = collection.get_worksheets()
    print(f"Split into {len(worksheets)} student worksheets")

    # Grade the worksheets
    print("Grading student worksheets...")
    student_answers, student_coords = grade_student_worksheets(
        collection,
        answer_choices,
        prediction_coords
    )

    # Create graded overlays with check and X assets
    print("Creating graded overlays with check and X assets...")
    graded_paths = create_graded_worksheets(
        collection,
        student_answers,
        answer_choices,
        student_coords,
        graded_dir,
        assets_dir
    )

    # Create flat worksheet structure
    print("Creating flat worksheet structure...")
    flat_paths = create_flat_worksheet_folder(
        collection,
        flat_dir
    )

    # Print statistics
    print("\n--- GRADING RESULTS ---")
    print(f"Number of students: {len(student_answers)}")

    # Calculate scores for each student
    student_scores = []
    for student_idx, student in enumerate(student_answers):
        correct_count = 0
        total_count = 0

        for page_idx, page_answers in enumerate(student):
            key_page_idx = page_idx % len(answer_choices)
            key_page = answer_choices[key_page_idx]

            for ans_idx, (student_ans, key_ans) in enumerate(zip(page_answers, key_page)):
                if student_ans == key_ans and student_ans != "null":
                    correct_count += 1
                total_count += 1

        if total_count > 0:
            score_percent = (correct_count / total_count) * 100
            student_scores.append({
                'student_idx': student_idx,
                'correct': correct_count,
                'total': total_count,
                'percentage': score_percent
            })
            print(f"Student {student_idx+1}: {correct_count}/{total_count} correct ({score_percent:.1f}%)")

    # Collect preview images (with grading marks)
    print("\n--- CREATING PDFs ---")
    preview_images = []
    for student_idx in range(len(student_answers)):
        for page_idx in range(len(student_answers[student_idx])):
            preview_filename = f"preview_student_{student_idx+1:03d}_page_{page_idx+1:03d}.png"
            preview_path = os.path.join(graded_dir, preview_filename)
            if os.path.exists(preview_path):
                preview_images.append(preview_path)

    # Convert preview images to PDF (these are the composited images with original + overlay)
    graded_pdf_created = False
    if preview_images:
        print(f"Creating graded worksheets PDF with {len(preview_images)} pages...")
        try:
            pdf_path = combine_images_to_pdf(
                preview_images,
                graded_pdf_path,
                target_aspect_ratio=target_aspect_ratio,
                dpi=300,
                crop_mode="center"
            )
            print(f"Graded worksheets PDF created: {pdf_path}")
            graded_pdf_created = True
        except Exception as e:
            print(f"Error creating graded worksheets PDF: {e}")
            # Try the reportlab fallback if img2pdf fails
            try:
                pdf_path = combine_images_to_pdf_reportlab(
                    preview_images,
                    graded_pdf_path,
                    target_aspect_ratio=target_aspect_ratio,
                    dpi=300,
                    crop_mode="center"
                )
                print(f"Graded worksheets PDF created with fallback method: {pdf_path}")
                graded_pdf_created = True
            except Exception as e2:
                print(f"Fallback method also failed: {e2}")

    # Convert overlay-only images to PDF (just the checkmarks and X's)
    overlay_pdf_created = False
    if graded_paths:
        print(f"Creating overlay-only PDF with {len(graded_paths)} pages...")

        # For transparent overlays, we need to create RGB images with white background
        overlay_with_bg_paths = []
        temp_files_to_clean = []

        try:
            # Create temporary images with white background
            for i, overlay_path in enumerate(graded_paths):
                # Open the overlay image (transparent PNG with just marks)
                with Image.open(overlay_path) as overlay_img:
                    # Create a new white RGB image
                    white_bg = Image.new("RGB", overlay_img.size, (255, 255, 255))

                    # Make sure overlay is in RGBA mode to get the alpha channel
                    if overlay_img.mode != "RGBA":
                        overlay_rgba = overlay_img.convert("RGBA")
                    else:
                        overlay_rgba = overlay_img

                    # Paste the overlay onto the white background using alpha as mask
                    white_bg.paste(overlay_rgba, (0, 0), overlay_rgba)

                    # Save as temporary file in JPEG format (no transparency)
                    temp_path = os.path.join(output_dir, f"temp_overlay_{i}.jpg")
                    white_bg.save(temp_path, "JPEG", quality=95)
                    overlay_with_bg_paths.append(temp_path)
                    temp_files_to_clean.append(temp_path)

                    print(f"  Processed overlay {i+1}/{len(graded_paths)}")

            # Create PDF from the overlay images with white background
            pdf_path = combine_images_to_pdf(
                overlay_with_bg_paths,
                overlay_pdf_path,
                target_aspect_ratio=target_aspect_ratio,
                dpi=300,
                crop_mode="center"
            )
            print(f"Overlay-only PDF created: {pdf_path}")
            overlay_pdf_created = True

        except Exception as e:
            print(f"Error creating overlay-only PDF: {e}")
            # Try the reportlab fallback
            try:
                # Create new white-background images for reportlab too
                reportlab_temp_paths = []
                for i, overlay_path in enumerate(graded_paths):
                    with Image.open(overlay_path) as overlay_img:
                        white_bg = Image.new("RGB", overlay_img.size, (255, 255, 255))
                        if overlay_img.mode != "RGBA":
                            overlay_rgba = overlay_img.convert("RGBA")
                        else:
                            overlay_rgba = overlay_img
                        white_bg.paste(overlay_rgba, (0, 0), overlay_rgba)
                        temp_path = os.path.join(output_dir, f"temp_reportlab_{i}.jpg")
                        white_bg.save(temp_path, "JPEG", quality=95)
                        reportlab_temp_paths.append(temp_path)
                        temp_files_to_clean.append(temp_path)

                pdf_path = combine_images_to_pdf_reportlab(
                    reportlab_temp_paths,
                    overlay_pdf_path,
                    target_aspect_ratio=target_aspect_ratio,
                    dpi=300,
                    crop_mode="center"
                )
                print(f"Overlay-only PDF created with fallback method: {pdf_path}")
                overlay_pdf_created = True
            except Exception as e2:
                print(f"Fallback method also failed: {e2}")

        finally:
            # Clean up temporary files
            for temp_file in temp_files_to_clean:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_file}: {e}")

    # Generate and save feedback
    feedback = {
        "summary": {
            "total_students": len(student_scores),
            "average_score": sum(s['percentage'] for s in student_scores) / len(student_scores)
        },
        "student_feedback": []
    }
    
    for student_idx, student in enumerate(student_answers):
        student_feedback = {
            "student_number": student_idx + 1,
            "questions": [],
            "score": next(s for s in student_scores if s['student_idx'] == student_idx)
        }

        question_number = 1
        
        for page_idx, page_answers in enumerate(student):
            key_page_idx = page_idx % len(answer_choices)
            key_page = answer_choices[key_page_idx]
            
            for q_idx, (student_ans, key_ans) in enumerate(zip(page_answers, key_page)):
                question_feedback = {
                    "question": question_number,
                    "student_answer": student_ans,
                    "correct_answer": key_ans,
                    "feedback": f"Question {question_number}: {'Correct!' if student_ans == key_ans else f'Incorrect. The correct answer is {key_ans}.'}"
                }
                student_feedback["questions"].append(question_feedback)
                question_number += 1
                    
        feedback["student_feedback"].append(student_feedback)
    
    # Save feedback to file
    with open(os.path.join(output_dir, "feedback.json"), 'w') as f:
        json.dump(feedback, f, indent=2)
    
    print("\nProcessing complete! Check the output directories for results.")

    # Return dictionary of results
    return {
        'graded_pdf': graded_pdf_path if graded_pdf_created else None,
        'overlay_pdf': overlay_pdf_path if overlay_pdf_created else None,
        'worksheets_dir': worksheets_dir,
        'graded_dir': graded_dir,
        'flat_dir': flat_dir,
        'student_answers': student_answers,
        'answer_key': answer_choices,
        'student_scores': student_scores,
        'feedback': feedback
    }

if __name__ == '__main__':
    # Set up paths
    rubric_path = "../uploads/answer_upload/answers.pdf"
    student_worksheets_path = "../uploads/student_work/students.pdf"
    output_dir = "../uploads/marked_up/"
    pages_per_student = 2
    assets_dir = "checkandx"

    # Process everything and print results
    results = process_grade_and_create_pdfs(
        rubric_path,
        student_worksheets_path,
        output_dir,
        pages_per_student,
        assets_dir
    )
    print("\nProcessing complete! Check the output directories for results.")