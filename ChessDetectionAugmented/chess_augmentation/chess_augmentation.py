import os
import glob
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define multiple transforms with different augmentations
transform1 = transforms.Compose([
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.Resize((736, 736)),  # Ensure the output size is fixed
    transforms.ToTensor()
])

transform2 = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.RandomAffine(30),
    transforms.Resize((736, 736)),  # Ensure the output size is fixed
    transforms.ToTensor()
])

transform3 = transforms.Compose([
    transforms.RandomGrayscale(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.Resize((736, 736)),  # Ensure the output size is fixed
    transforms.ToTensor()
])

# List of transforms to apply
transforms_list = [transform1, transform2, transform3]

# Path to the folder containing images
image_folder = r'C:\Users\talal\Downloads\JoVisionAITasks\ChessDetectionAugmented\Chess_dataset\old'

# Path to the folder where augmented images will be saved
augmented_folder = r'C:\Users\talal\Downloads\JoVisionAITasks\ChessDetectionAugmented\Chess_dataset\augmented'
os.makedirs(augmented_folder, exist_ok=True)

# Get list of all image files in the folder
image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))  # Change '*.jpg' to match your image file extension

# Iterate over each image, apply the transform, and save the augmented images
for image_path in image_paths:
    img = Image.open(image_path)
    base_name = os.path.basename(image_path).split('.')[0]
    
    # Apply each transform and save the augmented images
    for i, transform in enumerate(transforms_list):
        augmented_img = transform(img)
        
        # Convert the augmented image back to PIL format and save it
        augmented_img_pil = transforms.ToPILImage()(augmented_img)
        augmented_img_pil.save(os.path.join(augmented_folder, f'{base_name}_augmented_{i+1}.jpg'))

# Display a few augmented images
num_images_to_display = 4  # Number of images to display
augmented_image_paths = glob.glob(os.path.join(augmented_folder, '*.jpg'))
plt.figure(figsize=(10, 10))
for i in range(num_images_to_display):
    img = Image.open(augmented_image_paths[i])
    plt.subplot(2, 2, i + 1)
    plt.imshow(img)
    plt.axis('off')
plt.show()
