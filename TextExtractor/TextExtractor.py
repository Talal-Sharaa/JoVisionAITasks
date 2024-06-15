import sys
import cv2
import numpy as np
import easyocr
import os
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not open or find the image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    preprocessed_image_path = "preprocessed_"+os.path.splitext(os.path.basename(image_path))[0]+".png"
    cv2.imwrite(preprocessed_image_path, thresh)

    return preprocessed_image_path

def extract_text_from_image(image_path):
    try:
        preprocessed_image_path = preprocess_image(image_path)
        reader = easyocr.Reader(['en'], gpu=True)
        results = reader.readtext(preprocessed_image_path, detail=0)
        text = ' '.join(results)
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python TextExtractor.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    text = extract_text_from_image(image_path)
    print("Extracted Text:")
    print(text)
