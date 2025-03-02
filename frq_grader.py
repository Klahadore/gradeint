from PIL.JpegImagePlugin import APP
from google import genai
from google.genai import types
import httpx
import pathlib
import os
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile
import shutil
from typing import List

load_dotenv()
# This just takes in raw pdfs, split by answerkey, and an array of student worksheets
class GradeOutput(BaseModel):
    """Output schema for grading results."""
    score: float = Field(description="Numerical score based on the rubric")
    feedback: str = Field(description="Two sentences of constructive feedback for the student")


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GOOGLE_API_KEY")
    # other params...
)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)
def grade_worksheet(rubric_filepath, worksheet_filepath):
    rubric_loader = PyPDFLoader(str(rubric_filepath))
    rubric_pages = rubric_loader.load()
    rubric_content = "\n".join([page.page_content for page in rubric_pages])

    worksheet_loader = PyPDFLoader(worksheet_filepath)
    worksheet_pages = worksheet_loader.load()
    worksheet_content = "\n".join([page.page_content for page in worksheet_pages])

    prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an educational assessment expert.
                        Grade the student worksheet based on the provided rubric.
                        Provide a numerical score and two sentences of constructive feedback."""),
            ("human", """
            # Rubric:
            {rubric}

            # Student Worksheet:
            {student_worksheet}

            Grade this student worksheet based on the provided rubric.
            Return your response in JSON format with a numerical score based on the rubric
            and two sentences of constructive feedback for the student.
            """)
        ])
    structured_llm = model.with_structured_output(GradeOutput)
    formatted_messages = prompt.format_messages(
        rubric=rubric_content,
        student_worksheet=worksheet_content
    )
    result = structured_llm.invoke(formatted_messages)

    return result


import pathlib
import os
from pypdf import PdfReader, PdfWriter

def process_worksheets(whole_worksheets: str, output_dir: str, worksheet_len: int):
    """
    Split a large PDF into individual worksheets of a specified length.

    Args:
        whole_worksheets: Path to the combined PDF file containing all worksheets
        output_dir: Directory where individual worksheet PDFs will be saved
        worksheet_len: Number of pages in each worksheet

    Returns:
        List of paths to the individual worksheet PDFs
    """
    # Ensure output directory exists
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Open the combined PDF
    pdf_reader = PdfReader(whole_worksheets)
    total_pages = len(pdf_reader.pages)

    # Calculate number of worksheets
    num_worksheets = (total_pages + worksheet_len - 1) // worksheet_len  # Ceiling division

    output_files = []

    for worksheet_idx in range(num_worksheets):
        # Calculate page range for this worksheet
        start_page = worksheet_idx * worksheet_len
        end_page = min(start_page + worksheet_len, total_pages)

        # Create a new PDF for this worksheet
        pdf_writer = PdfWriter()

        # Add the worksheet's pages
        for page_num in range(start_page, end_page):
            pdf_writer.add_page(pdf_reader.pages[page_num])

        # Save the individual worksheet
        output_filename = f"worksheet_{worksheet_idx + 1}.pdf"
        output_file_path = output_path / output_filename

        with open(output_file_path, 'wb') as output_file:
            pdf_writer.write(output_file)

        output_files.append(str(output_file_path))

    return output_files

def grade_frqs(rubric_filepath: str, combined_worksheet_filepath: str, worksheet_len: int) -> List[GradeOutput]:
    """
    Grade multiple student worksheets by splitting a combined PDF and evaluating each against a rubric.

    Args:
        rubric_filepath: Path to the PDF containing the grading rubric
        combined_worksheet_filepath: Path to the PDF containing all student worksheets combined
        worksheet_len: Number of pages in each individual student worksheet

    Returns:
        List of GradeOutput objects containing scores and feedback for each worksheet, in order
    """
    # Create a temporary directory for the split worksheets
    temp_dir = tempfile.mkdtemp(prefix="worksheets_")

    try:
        # Split the combined worksheet file into individual worksheets
        individual_worksheets = process_worksheets(
            combined_worksheet_filepath,
            temp_dir,
            worksheet_len
        )

        # Grade each worksheet and collect results
        results = []
        for worksheet_path in individual_worksheets:
            try:
                # Grade the individual worksheet
                grade_result = grade_worksheet(rubric_filepath, worksheet_path)
                results.append(grade_result)
            except Exception as e:
                # In case of an error, create a placeholder result
                print(f"Error grading worksheet {worksheet_path}: {str(e)}")
                # Creating a default GradeOutput with error indicators
                error_grade = GradeOutput(
                    score=0.0,
                    feedback=f"Error processing this worksheet: {str(e)}"
                )
                results.append(error_grade)

        return results

    finally:
        # Clean up temporary directory and files
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary directory: {str(e)}")



from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
import textwrap
from pypdf import PdfReader

def create_grading_overlay(original_worksheet_path, output_path, grades, worksheet_length,
                           page_size=letter, margin_top=0.25, margin_left=0.25):
    """
    Create a PDF overlay with score and feedback annotations for each worksheet.

    Args:
        original_worksheet_path: Path to the original combined worksheet file
        output_path: Path where the overlay PDF will be saved
        grades: List of GradeOutput objects with score and feedback (from grade_frqs)
        worksheet_length: Number of pages per student worksheet
        page_size: PDF page size (default: letter)
        margin_top: Top margin in inches (default: 0.5)
        margin_left: Left margin in inches (default: 0.5)

    Returns:
        Path to the created overlay PDF
    """
    # Get information about the original PDF
    with open(original_worksheet_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        total_pages = len(pdf_reader.pages)

    # Create a new PDF with the same number of pages
    c = canvas.Canvas(output_path, pagesize=page_size)

    # Position for the annotations (top-left of each worksheet's first page)
    x = margin_left * inch
    y = page_size[1] - margin_top * inch  # Start from top of page

    # Define constants for box layout
    line_height = 15  # Height per line of text in points
    score_section_height = 30  # Height allocated for score in points
    padding_top = 5  # Padding at top of box in points
    padding_bottom = 5  # Padding at bottom of box in points
    box_width = 4 * inch  # Width of box

    # Create each page of the overlay
    for page_num in range(total_pages):
        # Determine if this is the first page of a worksheet
        worksheet_index = page_num // worksheet_length
        is_first_page = page_num % worksheet_length == 0

        # Only add annotations to the first page of each worksheet
        if is_first_page and worksheet_index < len(grades):
            grade = grades[worksheet_index]

            # Wrap feedback text to fit in the box
            wrapped_feedback = textwrap.fill(grade.feedback, width=75)
            feedback_lines = wrapped_feedback.split('\n')
            num_lines = len(feedback_lines)

            # Calculate dynamic box height based on number of lines
            feedback_height = num_lines * line_height
            box_height = score_section_height + feedback_height + padding_top + padding_bottom

            # Draw background box with dynamic height
            c.setFillColor(colors.white)
            c.setStrokeColor(colors.black)
            c.rect(x, y - box_height, box_width, box_height, fill=1, stroke=1)

            # Add the score at the top of the box
            score_text = f"Score: {grade.score}"
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(colors.blue)
            c.drawString(x + 10, y - (padding_top + 10), score_text)

            # Add the feedback lines
            c.setFont("Helvetica", 8)
            c.setFillColor(colors.black)

            for i, line in enumerate(feedback_lines):
                # Position each line below the score section
                line_y = y - (padding_top + score_section_height + (i * line_height))
                c.drawString(x + 10, line_y, line)

        # Move to the next page
        c.showPage()

    # Save the PDF
    c.save()

    return output_path


if __name__ == "__main__":
    rubric = "temp_files/aplangrubric.pdf"
    worksheet_file = "temp_files/aplang_response.pdf"
    output_dir = "temp_files/frq_worksheets"


    graded_frqs = grade_frqs(rubric, worksheet_file, 3)
    print(graded_frqs)

    output_overlay = "temp_files/lang_grades.pdf"
    overlay_path = create_grading_overlay(
        worksheet_file,
        output_overlay,
        graded_frqs,
        3
    )

    print(f"Created grading overlay at: {overlay_path}")
    print("Print this overlay on top of the original worksheets when running them through the printer again.")
