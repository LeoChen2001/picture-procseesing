# Chen
# 2023/6/13 22:43
import numpy as np
import cv2
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# Function to perform DPCM encoding
def dpcm_encode(image):
    height, width = image.shape
    encoded_image = np.zeros((height, width), dtype=np.float64)  # Use float64 for larger range

    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                predicted_value = 0
            elif i == 0:
                predicted_value = image[i, j-1]
            elif j == 0:
                predicted_value = image[i-1, j]
            else:
                predicted_value = image[i-1, j-1]

            prediction_error = np.clip(image[i, j] - predicted_value, 0, 255)
            encoded_image[i, j] = prediction_error

    return encoded_image

# Function to perform DPCM decoding
def dpcm_decode(encoded_image):
    height, width = encoded_image.shape
    decoded_image = np.zeros((height, width), dtype=np.float64)  # Use float64 for larger range

    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                predicted_value = 0
            elif i == 0:
                predicted_value = decoded_image[i, j-1]
            elif j == 0:
                predicted_value = decoded_image[i-1, j]
            else:
                predicted_value = decoded_image[i-1, j-1]

            decoded_image[i, j] = predicted_value + encoded_image[i, j]

    return decoded_image

# Function to quantize the image
def quantize_image(image, levels):
    if levels == 1:
        return image

    min_value = np.min(image)
    max_value = np.max(image)
    range_size = max_value - min_value
    step = range_size / (levels - 1)

    quantized_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(levels):
        threshold = min_value + i * step
        quantized_image[np.logical_and(image >= threshold, image < threshold + step)] = int(i * (255 / (levels - 1)))

    return quantized_image


# Load your own gray image
random_image = cv2.imread('w3.jpg', cv2.IMREAD_GRAYSCALE)

# Perform DPCM encoding and decoding with different quantizers
quantizers = [1, 2, 4, 8]
psnr_values = []
ssim_values = []

for quantizer in quantizers:
    encoded_image = dpcm_encode(random_image)
    quantized_image = quantize_image(encoded_image, quantizer)
    decoded_image = dpcm_decode(quantized_image)

    # Save reconstructed image
    cv2.imwrite(f'reconstructed_quantizer_{quantizer}.jpg', decoded_image)

    # Calculate PSNR and SSIM values
    psnr = compare_psnr(random_image, decoded_image, data_range=decoded_image.max() - decoded_image.min())
    ssim = compare_ssim(random_image, decoded_image)

    psnr_values.append(psnr)
    ssim_values.append(ssim)

    print(f"Quantizer: {quantizer}-bit")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print()

# Print PSNR and SSIM values for all quantizers
print("Quantizer\tPSNR (dB)\tSSIM")
for i, quantizer in enumerate(quantizers):
    print(f"{quantizer}-bit\t\t{psnr_values[i]:.2f}\t\t{ssim_values[i]:.4f}")