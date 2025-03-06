from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

# from mcq_grader import process_grade_and_create_pdfs
# import frq_grader

# import pikepdf

def count_pages(pdf_path):
    with pikepdf.open(pdf_path) as pdf:
        return len(pdf.pages)



app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store PDFs
UPLOAD_DIR = "../uploads/student_work/"
ANSWER_DIR = "../uploads/answer_upload/"
MARKED_DIR = "../uploads/marked_up/"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/grade-pdfs")
async def grade_pdfs_endpoint(
    assignmentType: str = Form(...),
    answer: UploadFile = File(...),
    worksheet: UploadFile = File(...)
):
    answer_path = os.path.join(ANSWER_DIR, f"answer_{answer.filename}")
    worksheet_path = os.path.join(UPLOAD_DIR, f"worksheet_{worksheet.filename}")
    graded_pdf_path = os.path.join(MARKED_DIR, "graded.pdf")  # Placeholder for the output file


    # Save Answer Key PDF
    with open(answer_path, "wb") as f:
        shutil.copyfileobj(answer.file, f)

    # Save Student Worksheet PDF
    with open(worksheet_path, "wb") as f:
        shutil.copyfileobj(worksheet.file, f)


    # pages_per_student = count_pages(answer_path)
    # if assignmentType == "mcq":
    #     output = process_grade_and_create_pdfs(answer_path, worksheet_path, graded_pdf_path, pages_per_student)

    # else:
    #     # 2) Use your "FRQ" code paths
    #     # E.g. open-ended answer detection, text extraction, LLM scoring, etc.
    #     pass

    # 📝 Simulate grading (In real use, call `grade_pdfs()` function)
    # shutil.copy(worksheet_path, graded_pdf_path)  # Just copying for now

    return {"message": "Files uploaded successfully", "graded_pdf": "/get-graded-pdf"}

# **New Endpoint: Serve the Graded PDF**
@app.get("/get-graded-pdf")
async def get_graded_pdf():
    graded_pdf_path = os.path.join(MARKED_DIR, "graded.pdf")

    # Ensure file exists before returning
    if not os.path.exists(graded_pdf_path):
        return {"error": "No graded PDF available"}

    return FileResponse(graded_pdf_path, media_type="application/pdf", filename="graded.pdf")

@app.get("/")
def root():
    return {"message": "PDF Grader API is running!"}
