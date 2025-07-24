import os
from PIL import Image
import numpy as np

class Captcha(object):
    def __init__(self, dataset_dir):
        # Initialize storage for character templates
        self.templates = []  # list of (char, binary_array)
        # Locate training data directory
        input_dir = os.path.join(dataset_dir, "input")
        output_dir = os.path.join(dataset_dir, "output")
        if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
            raise FileNotFoundError("Training data not found. Ensure 'sampleCaptchas/input' and 'sampleCaptchas/output' exist.")
        # Load all sample images and their corresponding text outputs
        for i in range(25):
            img_name = f"input{str(i).zfill(2)}.jpg"
            txt_name = f"output{str(i).zfill(2)}.txt"
            img_path = os.path.join(input_dir, img_name)
            txt_path = os.path.join(output_dir, txt_name)
            if not os.path.isfile(img_path) or not os.path.isfile(txt_path):
                # Skip if any expected file is missing (e.g., one missing label in sample set)
                continue
            # Read the known captcha text
            with open(txt_path, 'r') as f:
                captcha_text = f.read().strip()
            # Preprocess the image to binary
            image = Image.open(img_path).convert('L')             # grayscale
            img_array = np.array(image)
            binary = (img_array < 128).astype(np.uint8)           # threshold: 1 = black (text), 0 = white (background)
            # Segment the binary image into individual character sub-images
            segments = self._segment_columns(binary)
            # Verify we found the expected number of segments
            if len(segments) != len(captcha_text):
                # If segmentation doesn't match the text length, skip this sample
                continue
            # Extract each character region and store the template
            for (col_start, col_end), char in zip(segments, captcha_text):
                # Determine vertical bounds (top and bottom) for the character pixels
                rows = binary[:, col_start:col_end+1].sum(axis=1)
                top = np.argmax(rows > 0)                        # first row with a black pixel
                bottom = binary.shape[0] - np.argmax(np.flipud(rows) > 0) - 1  # last row with a black pixel
                # Crop the character region
                char_img = binary[top:bottom+1, col_start:col_end+1]
                self.templates.append((char, char_img))

    def __call__(self, im_path, save_path):
        """Load a captcha image and infer its text, saving the result to a file."""
        # Load and preprocess the image
        image = Image.open(im_path).convert('L')
        img_array = np.array(image)
        binary = (img_array < 128).astype(np.uint8)
        # Segment into character regions
        segments = self._segment_columns(binary)
        predicted_text = ""
        for (col_start, col_end) in segments:
            # Crop each character from the binary image
            rows = binary[:, col_start:col_end+1].sum(axis=1)
            top = np.argmax(rows > 0)
            bottom = binary.shape[0] - np.argmax(np.flipud(rows) > 0) - 1
            char_img = binary[top:bottom+1, col_start:col_end+1]
            # Identify the character by finding the best match among templates
            best_char = None
            best_score = float('inf')
            for template_char, template_img in self.templates:
                # To compare, align the template and char image by padding to same size
                h, w = char_img.shape
                ht, wt = template_img.shape
                H, W = max(h, ht), max(w, wt)
                # Create padded arrays (white background = 0)
                pad_char = np.zeros((H, W), dtype=np.uint8)
                pad_temp = np.zeros((H, W), dtype=np.uint8)
                # Place the binary images at top-left of the padding (they are already trimmed around the character)
                pad_char[:h, :w] = char_img
                pad_temp[:ht, :wt] = template_img
                # Compute difference (XOR) between the character and the template
                diff = np.sum(pad_char ^ pad_temp)
                if diff < best_score:
                    best_score = diff
                    best_char = template_char
            predicted_text += best_char if best_char is not None else "?"
        # Save the result to the output file
        with open(save_path, 'w') as out_file:
            out_file.write(predicted_text + "\n")

    def _segment_columns(self, binary_image):
        """Find continuous column regions (start and end indices) that contain character pixels."""
        col_sum = binary_image.sum(axis=0)
        segments = []
        in_char = False
        start_idx = 0
        for col, value in enumerate(col_sum):
            if value > 0 and not in_char:
                # start of a character segment
                in_char = True
                start_idx = col
            elif value == 0 and in_char:
                # end of a character segment
                end_idx = col - 1
                segments.append((start_idx, end_idx))
                in_char = False
        # If the last segment goes till the end of image
        if in_char:
            segments.append((start_idx, len(col_sum) - 1))
        return segments


if __name__ == '__main__':
    solver = Captcha(dataset_dir='sampleCaptchas')  # loads training data and builds the model
    solver("input100.jpg", "predicted_text.txt")
