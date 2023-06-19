# Chen
# 2023/6/19 10:20
import cv2
import numpy as np
import time

def otsu_threshold(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Normalize histogram
    hist_norm = hist.ravel() / hist.sum()

    # Calculate cumulative distribution
    cum_sum = np.cumsum(hist_norm)

    # Calculate cumulative mean
    cum_mean = np.cumsum(hist_norm * np.arange(256))

    # Calculate between-class variance
    between_var = cum_mean[-1] * cum_sum - cum_mean
    between_var = between_var ** 2 / (cum_sum * (1 - cum_sum))

    # Find optimal threshold
    threshold = np.argmax(between_var)

    # Apply threshold
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    return thresholded


def iterative_threshold(image, max_iter=100):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize threshold
    threshold = 128

    # Iterate until convergence or max iterations
    for _ in range(max_iter):
        # Split image into two regions based on threshold
        region1 = gray[gray <= threshold]
        region2 = gray[gray > threshold]

        # Calculate new threshold as the average of the means of the two regions
        new_threshold = 0.5 * (region1.mean() + region2.mean())

        # Check for convergence
        if abs(threshold - new_threshold) < 1e-5:
            break

        threshold = new_threshold

    # Apply threshold
    _, thresholded = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

    return thresholded


# Load image
image = cv2.imread('w5.jpg')

# Perform Otsu thresholding
start_time = time.time()
otsu_result = otsu_threshold(image)
otsu_time = time.time() - start_time

# Perform iterative thresholding
start_time = time.time()
iterative_result = iterative_threshold(image)
iterative_time = time.time() - start_time

# Compare execution times
print(f"Otsu method execution time: {otsu_time} seconds")
print(f"Iterative method execution time: {iterative_time} seconds")

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Otsu Result', otsu_result)
cv2.imshow('Iterative Result', iterative_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
