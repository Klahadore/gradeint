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
            contents=["What is circled? Answer with the capital answer choice letter. If there are multiple answers selected, return null.", img]
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
