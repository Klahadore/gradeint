from PIL import Image
from pathlib import Path

def process_raw_png(img: Image.Image) -> Image.Image:
    width, height = img.size
    aspect = width/height
    target_aspect = 1

    new_height = 1024
    new_width = int(1024 * aspect)

    resized_img = img.resize((new_width, new_height))

    paste_x = (1024 - new_width) // 2
    paste_y = (1024 - new_height) // 2

    padded_img = Image.new('1', (1024, 1024), 1)  # 1 is white in binary mode
    padded_img.paste(resized_img, (paste_x, paste_y))

    return padded_img

def process_png_directory(input_dir_path: str, output_dir_path: str):
    input_directory = Path(input_dir_path)
    output_directory = Path(output_dir_path)

        # Create output directory if it doesn't exist
    output_directory.mkdir(parents=True, exist_ok=True)

        # Collect ALL PNG files at the start and fix this list
    png_files = list(input_directory.glob("*.png"))

    print(f"Found {len(png_files)} PNG files")
    for file in png_files:
            # Get just the filename for the output path
        filename = file.name
        output_path = output_directory / filename

            # Use context manager to ensure proper file handling
        with Image.open(file) as img:
            processed = process_raw_png(img)
                # Save to the output directory, explicitly as PNG
            processed.save(output_path, format="PNG")
        print(f"Processed: {file} → {output_path}")

    print(f"Done. Processed {len(png_files)} files to {output_dir_path}")





if __name__ == "__main__":
    # img = Image.open("dataset/processed_pngs/frq_10_page_1.png")
    # img = process_raw_png(img)
    # print(img.size)
    # img.show()
    process_png_directory("dataset/processed_pngs/", "dataset/more_resized_pngs/")
