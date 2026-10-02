<<<<<<< HEAD
from PIL import Image, ImageFilter
import cv2
import numpy as np
import torchvision.transforms as transforms


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

=======
from PIL import Image
import torchvision.transforms as transforms


>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256


<<<<<<< HEAD
# ============================================================
# RESIZE IMAGE WITH PADDING
# ============================================================

=======
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
class ResizeWithPadding:

    def __init__(
        self,
        height=IMAGE_HEIGHT,
<<<<<<< HEAD
        width=IMAGE_WIDTH,
        mode="RGB"
    ):
        self.height = height
        self.width = width
        self.mode = mode

    def __call__(self, image):

        # Convert the image to the requested image mode
        image = image.convert(self.mode)

        # Get the original image dimensions
        original_width, original_height = image.size

        # Calculate the scaling factor while preserving
        # the original aspect ratio
=======
        width=IMAGE_WIDTH
    ):
        self.height = height
        self.width = width

    def __call__(self, image):

        # Always work with RGB
        image = image.convert("RGB")

        original_width, original_height = image.size

        # Calculate scale while preserving aspect ratio
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
        scale = min(
            self.width / original_width,
            self.height / original_height
        )

<<<<<<< HEAD
        # Calculate the new dimensions after scaling
=======
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
        new_width = max(
            1,
            int(original_width * scale)
        )

        new_height = max(
            1,
            int(original_height * scale)
        )

<<<<<<< HEAD
        # Resize the image without distorting the handwriting
=======
        # Resize without distorting handwriting
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

<<<<<<< HEAD
        # Create a white background canvas
        canvas = Image.new(
            self.mode,
=======
        # Create white background
        canvas = Image.new(
            "RGB",
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
            (self.width, self.height),
            "white"
        )

<<<<<<< HEAD
        # Calculate the position required to center
        # the resized image on the canvas
        x = (self.width - new_width) // 2
        y = (self.height - new_height) // 2

        # Paste the resized image onto the canvas
=======
        # Center the resized image
        x = (self.width - new_width) // 2
        y = (self.height - new_height) // 2

>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
        canvas.paste(image, (x, y))

        return canvas


<<<<<<< HEAD
# ============================================================
# BASELINE TRANSFORM
# ============================================================

def get_baseline_transform():

    transform = transforms.Compose([

        # Resize the image while preserving
        # its aspect ratio and adding padding
        ResizeWithPadding(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            mode="RGB"
        ),

        # Convert the PIL image into a PyTorch tensor
        transforms.ToTensor(),
    ])

    return transform


# ============================================================
# GRAYSCALE TRANSFORM
# ============================================================

def get_grayscale_transform():

    transform = transforms.Compose([

        # Convert the image to a single-channel grayscale image
        transforms.Grayscale(num_output_channels=1),

        # Resize while preserving aspect ratio
        # and adding white padding
        ResizeWithPadding(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            mode="L"
        ),

        # Convert the image into a PyTorch tensor
        transforms.ToTensor(),
    ])

    return transform


# ============================================================
# GRAYSCALE + DENOISING TRANSFORM
# ============================================================

def get_grayscale_denoise_transform():

    transform = transforms.Compose([

        # Convert the image to grayscale
        transforms.Grayscale(num_output_channels=1),

        # Apply a 3x3 median filter to reduce image noise
        transforms.Lambda(
            lambda image: image.filter(
                ImageFilter.MedianFilter(size=3)
            )
        ),

        # Resize while preserving aspect ratio
        # and adding white padding
        ResizeWithPadding(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            mode="L"
        ),

        # Convert the image into a PyTorch tensor
        transforms.ToTensor(),
    ])

    return transform


# ============================================================
# CLAHE IMAGE ENHANCEMENT
# ============================================================

def clahe_enhancement(image):
    """
    Apply CLAHE to a grayscale PIL image.
    Returns a grayscale PIL image.
    """

    # Convert the PIL image into a NumPy array
    image_array = np.array(image)

    # Create a CLAHE (Contrast Limited Adaptive
    # Histogram Equalization) object
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    # Apply CLAHE to enhance local image contrast
    enhanced = clahe.apply(image_array)

    # Convert the enhanced NumPy array back
    # into a grayscale PIL image
    return Image.fromarray(enhanced)


# ============================================================
# otsu Thresholding
# ============================================================
def otsu_threshold(image):
    """
    Apply Otsu's global thresholding to a grayscale PIL image.

    Returns:
        PIL.Image.Image: Binary grayscale image.
    """

    image_array = np.array(image)

    _, thresholded = cv2.threshold(
        image_array,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return Image.fromarray(thresholded)


# ============================================================
# GRAYSCALE + DENOISING + CLAHE TRANSFORM
# ============================================================

def get_grayscale_denoise_clahe_transform():

    transform = transforms.Compose([

        # Convert the image to grayscale
        transforms.Grayscale(
            num_output_channels=1
        ),

        # Apply a 3x3 median filter to reduce noise
        transforms.Lambda(
            lambda image: image.filter(
                ImageFilter.MedianFilter(size=3)
            )
        ),

        # Enhance local contrast using CLAHE
        transforms.Lambda(
            clahe_enhancement
        ),

        # Resize while preserving aspect ratio
        # and adding white padding
        ResizeWithPadding(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            mode="L"
        ),

        # Convert the processed image into a PyTorch tensor
        transforms.ToTensor(),
    ])

    return transform


# ============================================================
# GRAYSCALE + DENOISING + Thresholding TRANSFORM
# ============================================================
def get_grayscale_denoise_threshold_transform():

    transform = transforms.Compose([
        transforms.Grayscale(
            num_output_channels=1
        ),

        transforms.Lambda(
            lambda image: image.filter(
                ImageFilter.MedianFilter(size=3)
            )
        ),

        transforms.Lambda(
            otsu_threshold
        ),

        ResizeWithPadding(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            mode="L"
=======
def get_baseline_transform():

    transform = transforms.Compose([
        ResizeWithPadding(
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH
>>>>>>> 457675274f9874373bb904f7455bfb4e40a7e687
        ),

        transforms.ToTensor(),
    ])

    return transform