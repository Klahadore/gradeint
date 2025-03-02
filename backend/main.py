from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

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
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/grade-pdfs")
async def grade_pdfs_endpoint(
    answer: UploadFile = File(...), 
    worksheet: UploadFile = File(...)
):
    answer_path = os.path.join(UPLOAD_DIR, f"answer_{answer.filename}")
    worksheet_path = os.path.join(UPLOAD_DIR, f"worksheet_{worksheet.filename}")
    graded_pdf_path = os.path.join(UPLOAD_DIR, "graded.pdf")  # Placeholder for the output file

    # Save Answer Key PDF
    with open(answer_path, "wb") as f:
        shutil.copyfileobj(answer.file, f)

    # Save Student Worksheet PDF
    with open(worksheet_path, "wb") as f:
        shutil.copyfileobj(worksheet.file, f)

    # 📝 Simulate grading (In real use, call `grade_pdfs()` function)
    shutil.copy(worksheet_path, graded_pdf_path)  # Just copying for now

    return {"message": "Files uploaded successfully", "graded_pdf": "/get-graded-pdf"}

# **New Endpoint: Serve the Graded PDF**
@app.get("/get-graded-pdf")
async def get_graded_pdf():
    graded_pdf_path = os.path.join(UPLOAD_DIR, "graded.pdf")
    
    # Ensure file exists before returning
    if not os.path.exists(graded_pdf_path):
        return {"error": "No graded PDF available"}

    return FileResponse(graded_pdf_path, media_type="application/pdf", filename="graded.pdf")

@app.get("/")
def root():
    return {"message": "PDF Grader API is running!"}
