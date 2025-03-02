# import the inference-sdk
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import os
from dotenv import load_dotenv
from PIL import Image
from numpy import square
load_dotenv()

custom_configuration = InferenceConfiguration(confidence_threshold=0.09)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBO_KEY")
)
MODEL_ID = "my-first-project-6kwre/2"


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
def inference_on_img(image: Image.Image) -> list:
    result = CLIENT.infer(image, model_id=MODEL_ID)

    return result['predictions']


# takes in coordinates, and original pdf, and will return
# Takes in the coordinates from the original png converted image
def square_original_image(original_image: Image.Image):
    width, height = original_image.size
    padding_each_side = (height-width) // 2

    square_image = Image.new(mode=original_image.mode, size=(height, height), color="white")
    square_image.paste(original_image, (padding_each_side,0))

    return square_image


def draw_boxes_on_square(square_image: Image.Image):
    pass



if __name__ == "__main__":
    from utils import draw_roboflow_predictions
    img = Image.open("dataset/processed_pngs/frq_10_page_7.png")
    processed_img = process_raw_png(img)
    processed_img.show()
    print("processed,", processed_img.size)
    square_image = square_original_image(img)

    predictions = inference_on_img(processed_img)
    print(predictions)
    square_image = draw_roboflow_predictions(square_image, predictions)
    square_image.show()
