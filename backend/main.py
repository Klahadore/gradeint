from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from mcq_grader import process_grade_and_create_pdfs
import frq_grader
import pikepdf
import json

def count_pages(pdf_path):
    with pikepdf.open(pdf_path) as pdf:
        return len(pdf.pages)  # Added missing parenthesis

app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change to specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type", "Content-Length"],  # Add these headers
)

# Directory to store PDFs
UPLOAD_DIR = "../uploads/student_work/"
ANSWER_DIR = "../uploads/answer_upload/"
MARKED_DIR = "../uploads/marked_up/"

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/grade-pdfs")
def grade_pdfs_endpoint(
    assignmentType: str = Form(...),
    answer: UploadFile = File(...),
    worksheet: UploadFile = File(...)
):
    answer_path = os.path.join(ANSWER_DIR, f"answers.pdf")
    worksheet_path = os.path.join(UPLOAD_DIR, f"students.pdf")
    output_dir = MARKED_DIR  # Just pass the directory, let mcq_grader handle the rest

    # Save Answer Key PDF
    with open(answer_path, "wb") as f:
        shutil.copyfileobj(answer.file, f)

    # Save Student Worksheet PDF
    with open(worksheet_path, "wb") as f:
        shutil.copyfileobj(worksheet.file, f)

    pages_per_student = count_pages(answer_path)
    if assignmentType == "mcq":
        output = process_grade_and_create_pdfs(answer_path, worksheet_path, output_dir, pages_per_student)

    return {"message": "Files uploaded successfully"}

@app.get("/get-graded-pdf")
async def get_graded_pdf():
    graded_pdf_path = os.path.join(MARKED_DIR, "graded_worksheets.pdf")
    
    if not os.path.exists(graded_pdf_path):
        raise HTTPException(status_code=404, detail="No graded PDF available")
        
    return FileResponse(
        graded_pdf_path,
        media_type="application/pdf",
        filename="graded_worksheets.pdf",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Disposition": "inline; filename=graded_worksheets.pdf"
        }
    )

@app.get("/get-overlay-pdf")
async def get_overlay_pdf():
    overlay_pdf_path = os.path.join(MARKED_DIR, "overlay_only.pdf")
    
    if not os.path.exists(overlay_pdf_path):
        raise HTTPException(status_code=404, detail="No overlay PDF available")
        
    return FileResponse(
        overlay_pdf_path,
        media_type="application/pdf",
        filename="overlay_only.pdf",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Content-Disposition": "inline; filename=overlay_only.pdf"
        }
    )

@app.get("/get-feedback")
async def get_feedback():
    """Return AI feedback for the graded worksheets"""
    try:
        feedback_path = os.path.join(MARKED_DIR, "feedback.json")
        if not os.path.exists(feedback_path):
            # Return empty feedback structure instead of 404
            return {
                "summary": {
                    "total_students": 0,
                    "average_score": 0
                },
                "student_feedback": []
            }
            
        with open(feedback_path, 'r') as f:
            feedback = json.load(f)
            
        return feedback
    except Exception as e:
        print(f"Feedback error: {str(e)}")  # Debug print
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "PDF Grader API is running!"}
