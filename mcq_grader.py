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

                # Use coordinates directly - they're already in absolute pixel values
                center_x = bbox['x']
                center_y = bbox['y']
                width = bbox['width']
                height = bbox['height']

                print(f"  Center: ({center_x:.1f}, {center_y:.1f}), size: {width:.1f} x {height:.1f}")

                # Convert center coordinates to corner coordinates
                x1, y1, x2, y2 = prediction_to_box_coordinates(
                    bbox,
                    original_image.width,
                    original_image.height
                )

                print(f"  Box corners: ({x1}, {y1}) to ({x2}, {y2})")

                # Use top-left corner for placing marks
                mark_x = x1
                mark_y = y1

                print(f"  Placing mark at: ({mark_x}, {mark_y})")
                print(f"  Student answer: {student_answer}, Key answer: {key_answer}")

                # Determine if the answer is correct
                is_correct = (student_answer == key_answer) and (student_answer != "null")
                print(f"  Is correct: {is_correct}")

                # Use image assets if available, otherwise draw shapes
                if is_correct and check_img is not None:
                    # Paste checkmark at the top-left corner of the bounding box
                    graded_image.paste(check_img, (mark_x, mark_y), check_img)
                elif not is_correct and x_img is not None:
                    # Paste X at the top-left corner of the bounding box
                    graded_image.paste(x_img, (mark_x, mark_y), x_img)
                else:
                    # Fallback to drawing shapes if assets can't be loaded
                    draw = ImageDraw.Draw(graded_image)
                    mark_size = 64  # Fixed size to match asset size

                    if is_correct:
                        # Draw green checkmark
                        color = (0, 200, 0, 255)  # Green with full opacity
                        draw.line(
                            [
                                (mark_x, mark_y + mark_size/2),
                                (mark_x + mark_size/3, mark_y + mark_size),
                                (mark_x + mark_size, mark_y)
                            ],
                            fill=color,
                            width=max(3, mark_size//8)
                        )
                    else:
                        # Draw red X
                        color = (255, 0, 0, 255)  # Red with full opacity
                        draw.line(
                            [(mark_x, mark_y), (mark_x + mark_size, mark_y + mark_size)],
                            fill=color,
                            width=max(3, mark_size//8)
                        )
                        draw.line(
                            [(mark_x, mark_y + mark_size), (mark_x + mark_size, mark_y)],
                            fill=color,
                            width=max(3, mark_size//8)
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

# Updated process_and_grade_worksheets to include grading visuals


if __name__ == '__main__':
    # Set up paths
    rubric_path = "dataset/raw_pdf_dataset/mcq_89.pdf"
    student_worksheets_path = "dataset/raw_pdf_dataset/mcq_89.pdf"  # Using same file for testing
    output_dir = "temp_files/graded_worksheets_output"
    pages_per_student = 3  # Adjust based on your worksheet structure
    assets_dir = "checkandx"  # Directory containing check.png and x.png assets

    # Process the answer key
    print("Processing answer key...")
    answer_choices, prediction_coords = process_answer_key(rubric_path)
    print(f"Answer key has {len(answer_choices)} pages with answers")

    # Process the student worksheets
    print(f"Processing student worksheets from {student_worksheets_path}...")
    collection = WorksheetCollection()
    worksheets_dir = os.path.join(output_dir, "worksheets")
    process_worksheets_to_png(
        student_worksheets_path,
        worksheets_dir,
        pages_per_student,
        dpi=600,
        collection=collection
    )

    # Grade the worksheets
    print("Grading student worksheets...")
    student_answers, student_coords = grade_student_worksheets(
        collection,
        answer_choices,
        prediction_coords
    )

    # Create graded overlays with check and X assets
    print("Creating graded overlays with check and X assets...")
    graded_dir = os.path.join(output_dir, "graded_overlays")
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
    flat_dir = os.path.join(output_dir, "flat_worksheets")
    flat_paths = create_flat_worksheet_folder(
        collection,
        flat_dir
    )

    # Print statistics
    print("\n--- GRADING RESULTS ---")
    print(f"Number of students: {len(student_answers)}")

    # Print the first few answers from each student
    for i, student in enumerate(student_answers):
        if i < 3:  # Limit to first 3 students for brevity
            print(f"\nStudent {i+1}:")
            for j, page in enumerate(student):
                if j < 2:  # Limit to first 2 pages per student
                    print(f"  Page {j+1}: {page[:5]}...")  # Show first 5 answers

    # Print overlay image paths
    print("\n--- GENERATED FILES ---")
    print(f"Total graded overlays: {len(graded_paths)}")
    if graded_paths:
        print(f"First few overlay files:")
        for path in graded_paths[:3]:
            print(f"  {path}")

    print(f"\nTotal flat worksheet images: {len(flat_paths)}")
    if flat_paths:
        print(f"First few worksheet files:")
        for path in flat_paths[:3]:
            print(f"  {path}")

    # Calculate scores for each student
    print("\n--- STUDENT SCORES ---")
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
            print(f"Student {student_idx+1}: {correct_count}/{total_count} correct ({score_percent:.1f}%)")

    print("\nGrading complete! Check the output directories for results.")
