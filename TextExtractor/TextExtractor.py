import sys
from PIL import Image
import easyocr

def extract_text_from_image(image_path):
    try:
        reader = easyocr.Reader(['en'], gpu=True)
        img = Image.open(image_path)
        results = reader.readtext(image_path, detail=0)
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
