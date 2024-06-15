from PIL import Image
import sys
def color_to_black(img_path):
    # Load the image
    img = Image.open(img_path)
    img = img.convert("RGB")  # Ensure image is in RGB format

    # Get image dimensions
    width, height = img.size

    # Create a new image for the grayscale result
    grayscale_img = Image.new("L", (width, height))

    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            r, g, b = img.getpixel((x, y))

            # Calculate the grayscale value
            gray = int(0.2989 * r + 0.5870 * g + 0.1140 * b)

            # Set the grayscale value in the new image
            grayscale_img.putpixel((x, y), gray)

    return grayscale_img

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python color_to_black.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    grayscale_image = color_to_black(image_path)
    grayscale_image.show()
    grayscale_image.save("grayscale_image.jpg")