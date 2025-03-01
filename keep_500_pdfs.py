import os
import random

def delete_except_random(directory, num_to_keep):
    files = os.listdir(directory)
    if len(files) <= num_to_keep:
      return

    to_keep = random.sample(files, num_to_keep)

    for file in files:
        if file not in to_keep:
            file_path = os.path.join(directory, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

# Example usage
directory_path = "dataset/resized_pngs/"
num_files_to_keep = 500
delete_except_random(directory_path, num_files_to_keep)
