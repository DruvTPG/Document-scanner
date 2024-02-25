import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from scipy.ndimage import convolve
import cv2

# Load the image
image_path = r"C:\Users\druvt\OneDrive\Pictures\Druv photo.jpeg"
image = io.imread(image_path)

# Check if the image has an alpha channel (4 channels)
if image.shape[-1] == 4:
    # If it has an alpha channel, remove it
    image = image[:, :, :3]

# Convert the image to grayscale
def rgb_to_gray(image):
    # Check if the image is already grayscale
    if len(image.shape) == 2:
        return image

    # Luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])

    return gray_image

# Assuming 'image' is your RGB image (with or without an alpha channel)
# Convert it to grayscale
gray_image = rgb_to_gray(image)

# Ensure gray_image is a standard NumPy array
gray_image = np.array(gray_image)

# Define Sobel filters for horizontal and vertical edges
sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Apply the Sobel filters to the grayscale image using convolution
edges_horizontal = np.abs(convolve(gray_image, sobel_horizontal))
edges_vertical = np.abs(convolve(gray_image, sobel_vertical))

# Combine the horizontal and vertical edges
edges = np.sqrt(edges_horizontal**2 + edges_vertical**2)

# Normalize the edges to the range [0, 1]
edges = edges / np.max(edges)

# ... (previous code)

# Use contour detection to identify the document boundaries
contours, _ = cv2.findContours((edges * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Check if any contours were found
if contours:
    # Find the contour with the maximum area (presumed to be the document)
    max_contour = max(contours, key=cv2.contourArea)

    # Draw the contour on a black background
    contour_image = np.zeros_like(gray_image)
    cv2.drawContours(contour_image, [max_contour], -1, 255, thickness=cv2.FILLED)

    # Apply perspective transform to obtain a top-down view of the document
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    # Ensure approx has exactly four points
    if len(approx) == 4:
        approx = approx.reshape(4, 2)
    else:
        # Handle the case where the number of points is not four
        print("Warning: The number of approximated points is not four.")

    # Define the destination points for the perspective transformation
    dst_points = np.array([[0, 0], [800, 0], [800, 1000], [0, 1000]], dtype=np.float32)

    # Get the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(approx, dst_points)

    # Apply perspective transform to obtain a top-down view of the document
    transformed = cv2.warpPerspective(image, perspective_matrix, (800, 1000))

    # Plot the original image, detected edges, and the transformed document
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')

    plt.subplot(1, 3, 3)
    plt.imshow(transformed)
    plt.title('Scanned Document')

    plt.show()
else:
    print("No contours found.")
