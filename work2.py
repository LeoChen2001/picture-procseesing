# Chen
# 2023/6/12 10:28
import numpy as np
from scipy.signal import convolve2d
from scipy.fft import fft2, ifft2, fftshift

def wiener_filter_unknown_snr(image, psf, noise_var):
    H = fft2(psf, s=image.shape)
    F = fft2(image)
    G = H * F
    F_hat = np.abs(G) ** 2 / (np.abs(H) ** 2 * (np.abs(H) ** 2 + noise_var))
    image_hat = np.real(ifft2(F_hat * F))
    return image_hat

def wiener_filter_known_snr(image, psf, noise_var, snr):
    H = fft2(psf, s=image.shape)
    F = fft2(image)
    G = H * F
    F_hat = np.abs(G) ** 2 / (np.abs(H) ** 2 + noise_var / snr)
    image_hat = np.real(ifft2(F_hat * F))
    return image_hat

def wiener_filter_known_correlation(image, psf, noise_autocorr, image_autocorr):
    H = fft2(psf, s=image.shape)
    F = fft2(image)
    G = H * F
    denominator = np.abs(H) ** 2 * fftshift(noise_autocorr) + fftshift(image_autocorr)
    denominator[denominator == 0] = np.finfo(float).eps  # 将零元素替换为一个很小的非零值
    F_hat = np.conj(H) * G / denominator
    image_hat = np.real(ifft2(F_hat))
    return image_hat

# 示例使用
# 假设图像是一个5x5的矩阵，每个元素表示亮度值
image = np.array([[10, 20, 30, 40, 50],
                  [15, 25, 35, 45, 55],
                  [20, 30, 40, 50, 60],
                  [25, 35, 45, 55, 65],
                  [30, 40, 50, 60, 70]])

# 假设点扩散函数是一个5x5的矩阵
psf = np.array([[0.1, 0.2, 0.1, 0, 0],
                [0.2, 0.4, 0.2, 0, 0],
                [0.1, 0.2, 0.1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]])

# 假设噪声方差为10
noise_var = 10

# 假设信噪比为20
snr = 20

# 假设噪声自相关函数是一个5x5的矩阵
noise_autocorr = np.array([[1, 2, 1, 0, 0],
                           [2, 4, 2, 0, 0],
                           [1, 2, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

# 假设噪声自相关函数已进行频域调整
noise_autocorr_shifted = fftshift(noise_autocorr)

# 假设图像自相关函数
image_autocorr = np.array([[4, 8, 4, 0, 0],
                           [8, 16, 8, 0, 0],
                           [4, 8, 4, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])

# 假设图像自相关函数已进行频域调整
image_autocorr_shifted = fftshift(image_autocorr)

# 比较信噪比未知的复原结果
restored_image_unknown_snr = wiener_filter_unknown_snr(image, psf, noise_var)
print("信噪比未知的复原结果")
print(restored_image_unknown_snr)

# 比较信噪比已知的复原结果
restored_image_known_snr = wiener_filter_known_snr(image, psf, noise_var, snr)
print("信噪比已知的复原结果")
print(restored_image_known_snr)

# 比较图像和噪声自相关函数已知的复原结果
restored_image_known_correlation = wiener_filter_known_correlation(image, psf, noise_autocorr_shifted, image_autocorr_shifted)
print("图像和噪声自相关函数已知的复原结果")
print(restored_image_known_correlation)