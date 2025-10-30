#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备Colab训练数据集的脚本
将增强数据集打包成适合上传到Colab的压缩包
"""

import os
import zipfile
from pathlib import Path
import argparse

def create_colab_dataset_zip(dataset_path, output_path='augmented_dataset.zip', max_samples_per_class=150):
    """
    创建用于Colab训练的数据集压缩包
    
    Args:
        dataset_path: 增强数据集路径
        output_path: 输出压缩包路径
        max_samples_per_class: 每个类别最大样本数（控制压缩包大小）
    """
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    
    print(f"📦 开始创建Colab训练数据集压缩包...")
    print(f"📁 源路径: {dataset_path}")
    print(f"💾 输出: {output_path}")
    print(f"🎯 每类最大样本数: {max_samples_per_class}")
    
    total_files = 0
    total_classes = 0
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            
            if not os.path.isdir(class_path) or class_name.startswith('.'):
                continue
            
            # 获取该类别的所有图片
            images = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                images.extend(Path(class_path).glob(ext))
            
            if len(images) == 0:
                continue
            
            # 限制每个类别的样本数
            selected_images = list(images)[:max_samples_per_class]
            
            print(f"  📂 {class_name}: {len(selected_images)} 张图片")
            
            # 添加到压缩包
            for img_path in selected_images:
                # 在压缩包中的路径
                arc_path = f"augmented_dataset/{class_name}/{img_path.name}"
                zipf.write(str(img_path), arc_path)
                total_files += 1
            
            total_classes += 1
    
    # 检查压缩包大小
    zip_size_mb = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\n✅ 压缩包创建完成!")
    print(f"📊 统计信息:")
    print(f"  - 总类别数: {total_classes}")
    print(f"  - 总文件数: {total_files}")
    print(f"  - 压缩包大小: {zip_size_mb:.1f} MB")
    print(f"  - 保存位置: {os.path.abspath(output_path)}")
    
    if zip_size_mb > 100:
        print(f"⚠️  警告: 压缩包较大 ({zip_size_mb:.1f} MB)")
        print(f"   建议减少 max_samples_per_class 参数以降低大小")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='创建Colab训练数据集压缩包')
    parser.add_argument('--dataset', default='../augmented_dataset', help='增强数据集路径')
    parser.add_argument('--output', default='augmented_dataset_colab.zip', help='输出压缩包名称')
    parser.add_argument('--max_samples', type=int, default=120, help='每类最大样本数')
    
    args = parser.parse_args()
    
    success = create_colab_dataset_zip(
        dataset_path=args.dataset,
        output_path=args.output,
        max_samples_per_class=args.max_samples
    )
    
    if success:
        print(f"\n🚀 准备完成！")
        print(f"📋 Colab使用步骤:")
        print(f"   1. 将 {args.output} 上传到Colab")
        print(f"   2. 打开 colab_fish_training.ipynb")
        print(f"   3. 按顺序运行所有代码单元格")
        print(f"   4. 等待训练完成并下载模型")

if __name__ == "__main__":
    main()
