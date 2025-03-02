# import the inference-sdk
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import os
from dotenv import load_dotenv
from PIL import Image
from numpy import square
from utils import *
load_dotenv()

custom_configuration = InferenceConfiguration(confidence_threshold=0.2)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="vd8vIBbyTOqy6WETpLgy"
)

MODEL_ID = "my-first-project-6kwre/7"


def process_raw_png(img: Image.Image) -> Image.Image:
    # Convert image to black and white (binary)
    # First convert to grayscale
    grayscale_img = img.convert('L')

    # Then convert to binary with a threshold (128 is a common middle value)
    bw_img = grayscale_img.point(lambda x: 0 if x < 128 else 255, '1')

    # Get dimensions
    width, height = bw_img.size
    aspect = width/height

    # Calculate new dimensions maintaining aspect ratio
    new_height = 1024
    new_width = int(1024 * aspect)

    # Resize the black and white image
    resized_img = bw_img.resize((new_width, new_height))

    # Create padded square image
    paste_x = (1024 - new_width) // 2
    paste_y = (1024 - new_height) // 2

    padded_img = Image.new('1', (1024, 1024), 1)  # 1 is white in binary mode
    padded_img.paste(resized_img, (paste_x, paste_y))

    return padded_img


# take in pdf filepath, output PIL Image that is resized, scaled, etc
# takes in multiple page pdf

# runs yolo inference on PIL image, returns coordinates of boxes.
def inference_on_img(image: Image.Image, orig_image_size: tuple[int, int]) -> list:
    result = CLIENT.infer(image, model_id=MODEL_ID)
    orig_predictions = result['predictions']
    scaled_predictions = scale_predictions_to_resolution(orig_predictions, orig_image_size[1])

    return scaled_predictions


# takes in coordinates, and original pdf, and will return
# Takes in the coordinates from the original png converted image
def square_original_image(original_image: Image.Image):
    width, height = original_image.size
    padding_each_side = (height-width) // 2

    square_image = Image.new(mode=original_image.mode, size=(height, height), color="white")
    square_image.paste(original_image, (padding_each_side,0))

    return square_image

def extract_image_slices(square_image: Image.Image, predictions: list[dict]) -> list[Image.Image]:
    sorted_predictions = sorted(predictions, key=lambda pred: pred.get("y", 0))

    images = []
    for pred in sorted_predictions:
        x1, y1, x2, y2 = prediction_to_box_coordinates(pred, *square_image.size)
        cropped_image = square_image.crop((x1, y1, x2, y2))
        images.append(cropped_image)
    return images



if __name__ == "__main__":
    from utils import draw_roboflow_predictions
    img = Image.open("dataset/third_data_pngs/ap-english-language-and-composition-course-description - AP Lang Multi_page_16.png")
    processed_img = process_raw_png(img)
    processed_img.show()
    print("processed,", processed_img.size)
    square_image = square_original_image(img)

    predictions = inference_on_img(processed_img, square_image.size)
    print(predictions)
   # square_image = draw_roboflow_predictions(square_image, predictions)

    images = extract_image_slices(square_image, predictions)
    images[0].show()
