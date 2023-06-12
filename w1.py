# Chen
# 2023/6/11 9:27
from PIL import Image
import numpy as np

# 读取图像
image_path = "w1.jpg"
image = Image.open(image_path).convert("L")  # 转换为灰度图像
original_array = np.array(image)  # 将图像转换为NumPy数组

# 线性点运算
linear_array = original_array * 1.5  # 乘以一个常数来增加亮度
linear_array = np.clip(linear_array, 0, 255).astype(np.uint8)  # 剪切像素值到合理范围


# 分段线性点运算
segmented_array = original_array.copy()
segmented_array[original_array < 128] = (segmented_array[original_array < 128] * 0.5).astype(np.uint8)
segmented_array[original_array >= 128] = (segmented_array[original_array >= 128] * 2).astype(np.uint8)

# 非线性点运算（使用平方根函数）
nonlinear_array = np.sqrt(original_array) * 15  # 通过平方根函数增加对比度
nonlinear_array = np.clip(nonlinear_array, 0, 255).astype(np.uint8)

# 创建处理后的图像
linear_image = Image.fromarray(linear_array)
segmented_image = Image.fromarray(segmented_array)
nonlinear_image = Image.fromarray(nonlinear_array)


# 显示原图和处理后的图像
image.show()
linear_image.show()
segmented_image.show()
nonlinear_image.show()

