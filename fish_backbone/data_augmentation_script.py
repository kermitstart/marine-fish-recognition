#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
海洋鱼类数据集增强脚本
用于生成增强后的训练数据集，提升模型性能
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import glob
import shutil
from pathlib import Path
import argparse

def apply_augmentations(image, aug_type="all"):
    """
    对单张图片应用数据增强
    
    Args:
        image: PIL Image对象
        aug_type: 增强类型 ("rotation", "flip", "color", "brightness", "contrast", "all")
    
    Returns:
        augmented_image: 增强后的PIL Image对象
    """
    augmented = image.copy()
    
    if aug_type == "rotation":
        # 随机旋转 (-15 to 15 degrees)
        angle = random.uniform(-15, 15)
        augmented = augmented.rotate(angle, expand=True, fillcolor=(255, 255, 255))
    
    elif aug_type == "flip":
        # 随机水平翻转
        if random.random() > 0.5:
            augmented = augmented.transpose(Image.FLIP_LEFT_RIGHT)
    
    elif aug_type == "brightness":
        # 亮度调整
        enhancer = ImageEnhance.Brightness(augmented)
        factor = random.uniform(0.7, 1.3)
        augmented = enhancer.enhance(factor)
    
    elif aug_type == "contrast":
        # 对比度调整
        enhancer = ImageEnhance.Contrast(augmented)
        factor = random.uniform(0.8, 1.2)
        augmented = enhancer.enhance(factor)
    
    elif aug_type == "color":
        # 色彩饱和度调整
        enhancer = ImageEnhance.Color(augmented)
        factor = random.uniform(0.8, 1.2)
        augmented = enhancer.enhance(factor)
    
    return augmented

def generate_augmented_dataset(
    source_dir="../../dataset", 
    output_dir="../../augmented_dataset",
    target_samples_per_class=300,
    augmentations_per_image=3
):
    """
    生成增强数据集
    
    Args:
        source_dir: 原始数据集目录
        output_dir: 输出目录
        target_samples_per_class: 每个类别的目标样本数
        augmentations_per_image: 每张原图生成的增强图数量
    """
    
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        return None
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有类别
    classes = []
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        if os.path.isdir(class_dir) and not class_name.startswith('.'):
            classes.append(class_name)
    
    print(f"🚀 开始生成增强数据集...")
    print(f"📁 源目录: {source_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"📊 发现 {len(classes)} 个类别")
    print(f"🎯 目标: 每类 {target_samples_per_class} 张图片")
    print(f"🔄 每张原图生成 {augmentations_per_image} 张增强图")
    
    total_generated = 0
    
    for class_name in classes:
        print(f"\n📂 处理类别: {class_name}")
        
        # 创建类别输出目录
        class_output_dir = os.path.join(output_dir, class_name)
        Path(class_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取原始图片
        class_input_dir = os.path.join(source_dir, class_name)
        original_images = glob.glob(os.path.join(class_input_dir, "*.png")) + \
                         glob.glob(os.path.join(class_input_dir, "*.jpg")) + \
                         glob.glob(os.path.join(class_input_dir, "*.jpeg"))
        
        if len(original_images) == 0:
            print(f"  ⚠️ 跳过 {class_name}，没有找到图片")
            continue
        
        print(f"  📊 原始图片: {len(original_images)} 张")
        
        generated_count = 0
        
        # 首先复制原始图片
        for i, img_path in enumerate(original_images):
            if generated_count >= target_samples_per_class:
                break
                
            try:
                # 复制原图
                img_name = f"{class_name}_original_{i:04d}.png"
                output_path = os.path.join(class_output_dir, img_name)
                
                original_img = Image.open(img_path).convert('RGB')
                original_img.save(output_path)
                generated_count += 1
                
                # 生成增强图片
                for aug_idx in range(augmentations_per_image):
                    if generated_count >= target_samples_per_class:
                        break
                    
                    # 随机选择增强策略
                    aug_strategies = ['rotation', 'flip', 'brightness', 'contrast', 'color']
                    selected_aug = random.choice(aug_strategies)
                    
                    # 应用增强
                    augmented_img = apply_augmentations(original_img, selected_aug)
                    
                    # 保存增强图片
                    aug_img_name = f"{class_name}_aug_{selected_aug}_{i:04d}_{aug_idx:02d}.png"
                    aug_output_path = os.path.join(class_output_dir, aug_img_name)
                    augmented_img.save(aug_output_path)
                    generated_count += 1
                    
            except Exception as e:
                print(f"    ⚠️ 处理图片 {img_path} 时出错: {e}")
                continue
        
        print(f"  ✅ 生成图片: {generated_count} 张")
        total_generated += generated_count
    
    print(f"\n🎉 数据增强完成!")
    print(f"📊 总计生成: {total_generated} 张图片")
    print(f"📁 保存位置: {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='海洋鱼类数据集增强脚本')
    parser.add_argument('--source', default='../../dataset', help='源数据集目录')
    parser.add_argument('--output', default='../../augmented_dataset', help='输出目录')
    parser.add_argument('--target', type=int, default=200, help='每类目标样本数')
    parser.add_argument('--aug_per_img', type=int, default=2, help='每张原图生成的增强图数量')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    
    # 执行数据增强
    result_dir = generate_augmented_dataset(
        source_dir=args.source,
        output_dir=args.output,
        target_samples_per_class=args.target,
        augmentations_per_image=args.aug_per_img
    )
    
    if result_dir and os.path.exists(result_dir):
        print("\n📈 增强数据集统计:")
        for class_name in os.listdir(result_dir):
            class_dir = os.path.join(result_dir, class_name)
            if os.path.isdir(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {class_name}: {count} 张图片")

if __name__ == "__main__":
    main()
