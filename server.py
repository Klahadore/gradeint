from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up directories
UPLOADS_DIR = "uploads"
STUDENT_WORK_DIR = os.path.join(UPLOADS_DIR, "student_work")
MARKED_UP_DIR = os.path.join(UPLOADS_DIR, "marked_up")

# Create directories if they don't exist
for dir_path in [STUDENT_WORK_DIR, MARKED_UP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print("Server starting...")
print(f"Student work directory: {STUDENT_WORK_DIR}")
print(f"Marked up directory: {MARKED_UP_DIR}")

@app.post("/api/upload/student_work")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        filename = f"{uuid.uuid4()}.pdf"
        student_file_path = os.path.join(STUDENT_WORK_DIR, filename)
        marked_up_path = os.path.join(MARKED_UP_DIR, filename)

        # Save to student_work directory
        with open(student_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Make a copy in marked_up directory
        shutil.copy2(student_file_path, marked_up_path)
        
        print(f"File saved to: {student_file_path}")
        print(f"Copy saved to: {marked_up_path}")
        
        # Return the absolute path to the marked_up file
        abs_marked_up_path = os.path.abspath(marked_up_path)
        return JSONResponse({
            "status": "success",
            "filename": filename,
            "filepath": abs_marked_up_path  # Return absolute path
        })
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-pdf/{filename}")
async def process_pdf(filename: str):
    try:
        # Construct paths using existing directories
        source_path = os.path.join(STUDENT_WORK_DIR, filename)
        marked_up_path = os.path.join(MARKED_UP_DIR, filename)

        print(f"\nProcessing PDF: {filename}")
        print(f"Source: {source_path}")
        print(f"Destination: {marked_up_path}")

        # Copy the file to marked_up directory
        shutil.copy2(source_path, marked_up_path)
        print("File copied successfully")

        return JSONResponse({
            "status": "success",
            "filename": filename,
            "marked_up_path": marked_up_path
        })
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/marked-up/{filename}")
async def get_marked_up_pdf(filename: str):
    file_path = os.path.join(MARKED_UP_DIR, filename)
    if not os.path.exists(file_path):
        print(f"PDF not found at: {file_path}")
        raise HTTPException(status_code=404, detail="PDF not found")
    
    print(f"Serving PDF from: {file_path}")
    return FileResponse(
        path=file_path,
        media_type='application/pdf',
        filename=filename
    )

# Serve frontend files
app.mount("/", StaticFiles(directory="frontend/src", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
