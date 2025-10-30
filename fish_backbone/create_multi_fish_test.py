"""
创建多鱼测试图片
"""

import cv2
import numpy as np
import os
from PIL import Image

def create_multi_fish_test_image():
    """创建包含多条鱼的测试图片"""
    
    # 从数据集中选择几张鱼的图片
    fish_images = [
        "/Users/wen11/MarineFish_recognition/dataset_processed/train/Chaetodon_lunulatus/Chaetodon_lunulatus_0000.png",
        "/Users/wen11/MarineFish_recognition/dataset_processed/train/Chromis_chrysura/Chromis_chrysura_0000.png",
        "/Users/wen11/MarineFish_recognition/dataset_processed/train/Amphiprion_clarkia/Amphiprion_clarkia_0000.png",
    ]
    
    # 创建一个较大的背景图像 (蓝色海洋背景)
    canvas_width = 600
    canvas_height = 400
    canvas = np.full((canvas_height, canvas_width, 3), (139, 69, 19), dtype=np.uint8)  # 海洋蓝色
    
    loaded_images = []
    
    # 加载和调整鱼类图片
    for fish_path in fish_images:
        if os.path.exists(fish_path):
            fish_img = cv2.imread(fish_path)
            if fish_img is not None:
                # 调整大小
                fish_img = cv2.resize(fish_img, (120, 120))
                loaded_images.append(fish_img)
    
    if not loaded_images:
        print("❌ 未找到测试图片")
        return None
    
    # 在画布上放置多条鱼
    positions = [
        (50, 50),      # 左上
        (300, 80),     # 右上
        (150, 220),    # 中下
        (450, 180),    # 右下
    ]
    
    for i, fish_img in enumerate(loaded_images[:4]):  # 最多4条鱼
        if i < len(positions):
            x, y = positions[i]
            # 确保不超出边界
            if x + fish_img.shape[1] <= canvas_width and y + fish_img.shape[0] <= canvas_height:
                canvas[y:y+fish_img.shape[0], x:x+fish_img.shape[1]] = fish_img
    
    # 保存合成图片
    output_path = "./uploads/multi_fish_test.png"
    cv2.imwrite(output_path, canvas)
    print(f"✅ 多鱼测试图片已保存: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_multi_fish_test_image()
