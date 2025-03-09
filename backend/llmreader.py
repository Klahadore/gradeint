import os
from dotenv import load_dotenv
from google import genai
import PIL.Image

# Load API key from environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {bool(google_api_key)}") 

# Initialize Google Gemini client
client = genai.Client(api_key=google_api_key)

def scan_images(images, valid_choices=None):
    """
    Scans a list of PIL Image objects and extracts the circled answers.
    
    Parameters:
        images (list): A list of PIL.Image objects to process.
        valid_choices (list, optional): List of valid answer choices (e.g., ['A', 'B', 'C', 'D', 'E']).
                                       If None, defaults to A-E.

    Returns:
        list: A list of extracted answers from the images.
    """
    model_responses = []
    
    # Set default valid choices if none provided
    if valid_choices is None:
        valid_choices = ['A', 'B', 'C', 'D', 'E']
    
    # Create dynamic prompt based on valid choices
    choices_str = ", ".join(valid_choices)
    prompt = f"Look carefully at the image and identify which answer choice ({choices_str}) has a clear circle or oval around it. Focus on these specific criteria:\n1. The marking must be a complete, closed circle or oval\n2. The marking must fully enclose exactly one answer choice\n3. The enclosed choice must be clearly readable\n4. There should be no other significant markings or circles\nAnswer with the capital letter of the circled choice. Return 'null' if:\n- Multiple answers are circled\n- The circle/oval is incomplete or unclear\n- No answer is clearly circled\n- The circled choice is not one of the valid options ({choices_str})\nBe very strict in applying these criteria."

    # Process each image
    for idx, img in enumerate(images):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, img]
        )

        extracted_text = response.text.strip()  # Remove extra whitespace
        
        # Validate the response is a valid choice
        if extracted_text not in valid_choices and extracted_text != 'null':
            extracted_text = 'null'
            
        model_responses.append(extracted_text)

        # Print each response for debugging
        print(f"Question {idx+1}: {extracted_text}")

    return model_responses  # Return the extracted answers as a list

def compare_answer_sets(student_answers, correct_answers):
    """
    Compares two sets of answers and calculates the match ratio.
    
    Parameters:
        student_answers (list): List of student's answers
        correct_answers (list): List of correct answers

    Returns:
        float: Ratio of matching answers (0.0 to 1.0)
        int: Number of matches
    """
    if len(student_answers) != len(correct_answers):
        raise ValueError("Answer sets must be of equal length")

    matches = sum(1 for i in range(len(student_answers)) 
                 if student_answers[i] == correct_answers[i])
    match_ratio = matches / len(student_answers)
    
    return match_ratio, matches

def scan_images(image_dir):
    """
    Scans all images in the specified directory and extracts the circled answers.
    
    Parameters:
        image_dir (str): The directory containing PNG images.

    Returns:
        list: A list of extracted answers from the images.
    """

    def extract_number(filename):
        """Extracts the numerical part of the filename for proper sorting."""
        return int(filename.split(".png")[0])  # Assumes filename format like '1.png'

    # Get and sort image files numerically
    image_files = sorted(
        [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")],
        key=lambda x: extract_number(os.path.basename(x))
    )

    images = [PIL.Image.open(img_path) for img_path in image_files]

    model_responses = []

    # Process each image
    for idx, img in enumerate(images):
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Look carefully at the image and identify which answer choice (A, B, C, D, or E) has a clear circle or oval around it. Focus on complete, closed circular or oval markings that fully enclose a single answer choice. Answer with the capital letter of the circled choice. If you see multiple circles, partial circles, or no clear circles, return 'null'. Be strict about requiring a complete circle or oval marking.", img]
        )

        extracted_text = response.text.strip()  # Remove extra whitespace
        model_responses.append(extracted_text)

        # Print each response for debugging
        print(f"Question {idx+1}: {extracted_text}")

    # Final output
    print("\nFinal Extracted Answers:")
    print(model_responses)

    return model_responses  # Return the extracted answers as a list

answers_set = scan_images("assets/answerpng")
examples_set = scan_images("assets/examplepng")

print("Set 1 Answers:", answers_set)
print("Set 2 Answers:", examples_set)

if len(answers_set) != len(examples_set):
    print("Error: The two sets have different lengths!")
else:
    matches = sum(1 for i in range(len(answers_set)) if answers_set[i] == examples_set[i])
    
    # Calculate match ratio
    match_ratio = matches / len(answers_set)
    
    # Print result
    print(f"Match Ratio: {match_ratio:.2f}")