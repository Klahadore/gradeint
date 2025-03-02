import os
from pdf2image import convert_from_path
from PIL import Image
import glob

def process_pdfs(input_folder, output_folder, dpi=300):
    """
    Process all PDFs in input_folder and save results in output_folder.

    Args:
        input_folder: Path to folder containing PDF files
        output_folder: Path to save the output PNGs
        dpi: Resolution for the PNG images (default 300)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all PDF files in the input folder
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

    for pdf_file in pdf_files:
        base_filename = os.path.basename(pdf_file)
        name_without_ext = os.path.splitext(base_filename)[0]

        print(f"Processing {base_filename}...")

        # Convert PDF to images
        images = convert_from_path(pdf_file, dpi=dpi)

        # Process each page
        for i, image in enumerate(images):
            # Convert to black and white (binary)
            bw_image = image.convert("1")  # Convert to 1-bit black and white

            # Save as PNG
            output_file = os.path.join(
                output_folder, f"{name_without_ext}_page_{i+1}.png"
            )
            bw_image.save(output_file)

            print(f"  Saved {output_file}")

    print("Processing complete!")

# Example usage
if __name__ == "__main__":
    input_folder = "dataset/third_data_pdfs"
    output_folder = "dataset/third_data_pngs"

    process_pdfs(input_folder, output_folder)
