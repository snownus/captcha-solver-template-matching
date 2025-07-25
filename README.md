# Simple Captcha Recognition Solver

This project implements a simple AI model to recognize fixed-format CAPTCHA images. It was developed as part of an AI technical test.

## Problem Overview

A website uses CAPTCHA images to prevent automated bots. Each CAPTCHA:

- Contains **exactly 5 characters**
- Uses consistent **font, size, spacing, background, and color**
- Has **no skew or distortion**
- Includes **uppercase A–Z and digits 0–9**

You are given a dataset of 25 sample CAPTCHA images covering all possible characters, along with their labels.

## Objective

Build a simple algorithm that can accurately infer the text from unseen CAPTCHA images using only basic understanding of the RGB color space (no advanced computer vision required).

## Solution Approach

The solution follows a three-step process:

1. **Preprocessing**

   - Convert the image to grayscale and binarize it using a fixed threshold.
   - This helps extract black text from a mostly white background.
2. **Segmentation**

   - Detect and extract 5 character regions from the binary image.
   - Based on consistent character width and spacing.
3. **Template Matching (Recognition)**

   - Each character is compared to pre-learned templates using pixel-wise difference.
   - The closest match is returned as the predicted character.

## Code Structure

- `Captcha` class:
  - `__init__`: Loads and builds templates from the 25 sample images.
  - `__call__(im_path, save_path)`: Takes an image path and outputs the predicted 5-character text to a file.

## Getting Started

### Prerequisites

- Python 3.7+
- Install required packages:

```bash
pip install numpy pillow
```

### Dataset Structure

Unzip the provided dataset so that the directory looks like this:

```
sampleCaptchas/ 
   input/ 
      input00.jpg 
      input01.jpg 
      ... 

   output/ 
      output00.txt 
      output01.txt 
      ... 
```

### Run the Inference

```bash
solver = Captcha()
solver("path_to/unseen_captcha.jpg", "output_prediction.txt")
```
This will save the predicted 5-character CAPTCHA string into output_prediction.txt. A practical example is shown in `__main__` function of `Captcha.py` and can run directly to test. 
