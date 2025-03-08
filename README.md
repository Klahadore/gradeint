# Gradient
The all-in-one grading platform for teachers

# HenHacks 2025 Winner 
🏆1 of 10 major category winner groups out of 130 projects🏆
2nd Place 🥈 for Productivity Category!

Hosted by the University of Delaware | MLH | JP Morgan Chase | M&T Tech | Labware | Teach for America | Berkley |  IWCS

# How it Works
Gradient takes in two files: an answer key PDF and a student work PDF. These files can either be uploaded directly through your computer's finder or through a connected printer. Once the files are uploaded, our custom computer vision model (trained with Roboflow) segments the two PDFs into individual question PNG images. Two PNG images are produced for each question: the answer key question PNG and the student work question PNG. Using Gemini 2.0 Flash and its built-in OCR, we feed these two PNG images into the LLM and determine if there is a circling discrepancy between what was answered by the student and what was answered by the answer key. After determining whether or not the student got the question correct, we overlay "checkmark" and "x" PNG images onto a blank PDF and onto a marked-up PDF, both of which can be printed. You can either print out and view the marked-up PDF or you can print out and view just the overlay. If you wish to print the "check marks" and "xs" over the student’s work, you can do so by feeding into the printer the student's original work and printing on top of the paper the PDF overlay. 

# Step-By-Step Instructions
1. Create a .env file containing your Google Gemini API key and your Roboflow API key.
   - Should be in the form ```ROBO_KEY='your_api_key'
     GOOGLE_API_KEY='your_api_key'```
3. Open the terminal and run ```conda create -n gradient ``` to create a VM.
4. Run ```conda activate gradient ``` to activate VM.
5. Run ```pip install -r requirements.txt``` to download all packages.
6. Run ```cd backend``` then ```uvicorn main:app --reload --port 8000```.
7. Open another terminal and activate conda VM. Run ```cd frontend``` then ```python -m http.server 3000```.
8. Enter the link ```http://[::]:3000/``` into your browser.

When files are moved around, you might have to change imports on some files. To install inferencesdk, specify the version to 2. something on install.



Todo to get this cleaned up:
1. All backend code moved to backend
2. Remove temporary files/directories
3. Normalize Dependencies 
4. Normalize and minimize frontend
5. API server should connect to ML stuff (filename issue RN)
6. Generate PYProject.toml
7. Update the readme
8. Update the DevPost
