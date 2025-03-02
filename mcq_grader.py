import os
from dotenv import load_dotenv
from google import genai
from PIL import Image
import asyncio
from utils import *
from yolo_inference import *

# Load API key from environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
print(f"API Key loaded: {bool(google_api_key)}")

# Initialize Google Gemini client
client = genai.Client(api_key=google_api_key)


async def scan_single_image(client, img, idx):
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
    print(f"Question {idx+1}: {extracted_text}")
    return extracted_text

async def scan_images(images: list[Image.Image], client):
    # Create tasks for all images
    tasks = [
        scan_single_image(client, img, idx)
        for idx, img in enumerate(images)
    ]

    # Run all tasks concurrently and wait for all to complete
    model_responses = await asyncio.gather(*tasks)

    # Final output
    print("\nFinal Extracted Answers:")
    print(model_responses)

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


if __name__ == '__main__':
    results = asyncio.run(main())
    results = clean_outputs(results)
    print(results)
# How to call this in your main code:
# results = asyncio.run(main(images, client))
 # Return the extracted answers as a list
