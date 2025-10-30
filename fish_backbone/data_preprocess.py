# 海洋鱼类数据集预处理和均衡化
# 实现数据均衡、划分和预处理功能

import os
import shutil
import random
import glob
from collections import Counter
import json

def analyze_dataset(dataset_root):
    """
    分析原始数据集的分布情况
    """
    print("🔍 分析原始数据集...")
    
    classes = [d for d in os.listdir(dataset_root) 
               if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')]
    classes.sort()
    
    class_stats = {}
    total_images = 0
    
    for cls in classes:
        cls_dir = os.path.join(dataset_root, cls)
        images = glob.glob(os.path.join(cls_dir, "*.png")) + \
                glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                glob.glob(os.path.join(cls_dir, "*.jpeg"))
        
        class_stats[cls] = len(images)
        total_images += len(images)
    
    # 统计分析
    counts = list(class_stats.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    
    print(f"\n📊 数据集分析结果:")
    print(f"  总类别数: {len(classes)}")
    print(f"  总图片数: {total_images:,}")
    print(f"  平均每类: {avg_count:.1f} 张")
    print(f"  最少样本: {min_count} 张 ({min(class_stats, key=class_stats.get)})")
    print(f"  最多样本: {max_count:,} 张 ({max(class_stats, key=class_stats.get)})")
    print(f"  数据不均衡比: {min_count/max_count:.4f}")
    
    # 详细分布
    print(f"\n📈 各类别样本数量:")
    for i, (cls, count) in enumerate(sorted(class_stats.items(), key=lambda x: x[1], reverse=True)):
        status = ""
        if count > 200:
            status = "📉 需要抽样"
        elif count < 50:
            status = "⚠️  样本较少" 
        elif count < 100:
            status = "🟡 中等样本"
        else:
            status = "✅ 样本充足"
            
        print(f"  {i+1:2d}. {cls:<30} {count:4d} 张 {status}")
    
    return classes, class_stats

def balance_dataset(original_root, balanced_root, max_samples_per_class=200, min_samples_per_class=20):
    """
    均衡化数据集 - 解决样本不均衡问题
    """
    print(f"\n🎯 开始数据均衡化处理...")
    print(f"  策略: 每类最多保留 {max_samples_per_class} 张图片")
    print(f"  过滤: 少于 {min_samples_per_class} 张的类别将被排除")
    
    # 创建均衡数据集目录
    if os.path.exists(balanced_root):
        print(f"  清理旧的均衡数据集: {balanced_root}")
        shutil.rmtree(balanced_root)
    os.makedirs(balanced_root)
    
    # 获取原始数据统计
    classes, class_stats = analyze_dataset(original_root)
    
    # 过滤和处理每个类别
    processed_stats = {}
    excluded_classes = []
    
    for cls in classes:
        original_count = class_stats[cls]
        
        # 跳过样本过少的类别
        if original_count < min_samples_per_class:
            excluded_classes.append((cls, original_count))
            print(f"  🚫 排除 {cls}: 样本数 {original_count} < {min_samples_per_class}")
            continue
        
        # 创建类别目录
        cls_balanced_dir = os.path.join(balanced_root, cls)
        os.makedirs(cls_balanced_dir)
        
        # 获取该类别的所有图片
        cls_original_dir = os.path.join(original_root, cls)
        all_images = glob.glob(os.path.join(cls_original_dir, "*.png")) + \
                    glob.glob(os.path.join(cls_original_dir, "*.jpg")) + \
                    glob.glob(os.path.join(cls_original_dir, "*.jpeg"))
        
        # 随机抽样或全部保留
        if len(all_images) > max_samples_per_class:
            # 随机抽样
            random.shuffle(all_images)
            selected_images = all_images[:max_samples_per_class]
            action = f"抽样 {len(selected_images)}/{len(all_images)}"
        else:
            # 全部保留
            selected_images = all_images
            action = f"全保留 {len(selected_images)}"
        
        # 复制选中的图片
        for i, img_path in enumerate(selected_images):
            filename = os.path.basename(img_path)
            # 重命名以避免冲突
            name, ext = os.path.splitext(filename)
            new_filename = f"{cls}_{i:04d}{ext}"
            new_path = os.path.join(cls_balanced_dir, new_filename)
            shutil.copy2(img_path, new_path)
        
        processed_stats[cls] = len(selected_images)
        print(f"  ✅ {cls:<30} {action}")
    
    # 输出处理结果
    print(f"\n✅ 数据均衡化完成!")
    print(f"  保留类别: {len(processed_stats)} 个")
    print(f"  排除类别: {len(excluded_classes)} 个")
    
    if excluded_classes:
        print(f"\n  被排除的类别:")
        for cls, count in excluded_classes:
            print(f"    - {cls}: {count} 张")
    
    # 新数据集统计
    total_balanced = sum(processed_stats.values())
    avg_balanced = total_balanced / len(processed_stats)
    min_balanced = min(processed_stats.values())
    max_balanced = max(processed_stats.values())
    
    print(f"\n📊 均衡后数据集统计:")
    print(f"  总类别数: {len(processed_stats)}")
    print(f"  总图片数: {total_balanced:,}")
    print(f"  平均每类: {avg_balanced:.1f} 张")
    print(f"  样本范围: {min_balanced} - {max_balanced} 张")
    print(f"  新均衡比: {min_balanced/max_balanced:.4f}")
    
    return processed_stats

def split_balanced_dataset(balanced_root, output_root, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将均衡化的数据集划分为训练集、验证集和测试集
    """
    print(f"\n📂 划分数据集...")
    print(f"  比例 - 训练:{train_ratio:.0%} 验证:{val_ratio:.0%} 测试:{test_ratio:.0%}")
    
    # 验证比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("训练、验证、测试比例之和必须等于1")
    
    # 创建输出目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(output_root, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(split_dir)
    
    # 获取所有类别
    classes = [d for d in os.listdir(balanced_root) 
               if os.path.isdir(os.path.join(balanced_root, d))]
    
    split_stats = {'train': {}, 'val': {}, 'test': {}}
    
    for cls in classes:
        cls_dir = os.path.join(balanced_root, cls)
        all_images = glob.glob(os.path.join(cls_dir, "*.png")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpeg"))
        
        # 随机打散
        random.shuffle(all_images)
        
        total_count = len(all_images)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        # test_count = total_count - train_count - val_count  # 剩余的都给测试集
        
        # 划分数据
        splits_data = {
            'train': all_images[:train_count],
            'val': all_images[train_count:train_count + val_count],
            'test': all_images[train_count + val_count:]
        }
        
        # 创建各split的类别目录并复制文件
        for split, images in splits_data.items():
            split_cls_dir = os.path.join(output_root, split, cls)
            os.makedirs(split_cls_dir, exist_ok=True)
            
            for img_path in images:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(split_cls_dir, filename))
            
            split_stats[split][cls] = len(images)
        
        print(f"  {cls:<30} 训练:{len(splits_data['train']):3d} 验证:{len(splits_data['val']):3d} 测试:{len(splits_data['test']):3d}")
    
    # 输出统计信息
    print(f"\n✅ 数据集划分完成!")
    for split in splits:
        total = sum(split_stats[split].values())
        print(f"  {split.upper():>5}集: {total:4d} 张图片, {len(split_stats[split])} 个类别")
    
    return split_stats

def generate_paddlex_format(dataset_root, output_dir):
    """
    生成PaddleX训练所需的格式文件
    """
    print(f"\n📝 生成PaddleX格式文件...")
    
    # 获取类别列表
    train_dir = os.path.join(dataset_root, 'train')
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"  发现 {len(classes)} 个类别")
    
    # 创建标签映射
    label_map = {cls: i for i, cls in enumerate(classes)}
    
    # 生成各种格式文件
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            continue
            
        data_list = []
        
        for cls in classes:
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.exists(cls_dir):
                continue
                
            images = glob.glob(os.path.join(cls_dir, "*.png")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpeg"))
            
            for img_path in images:
                # 相对于dataset_root的路径
                rel_path = os.path.relpath(img_path, dataset_root)
                data_list.append(f"{rel_path}\t{label_map[cls]}")
        
        # 保存列表文件
        list_file = os.path.join(output_dir, f"{split}_list.txt")
        with open(list_file, 'w', encoding='utf-8') as f:
            for item in data_list:
                f.write(f"{item}\n")
        
        print(f"  ✅ {split}_list.txt: {len(data_list)} 条记录")
    
    # 保存标签文件
    label_file = os.path.join(output_dir, "label_list.txt")
    with open(label_file, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    # 保存类别名称文件
    class_names_file = os.path.join(output_dir, "class_names.txt")
    with open(class_names_file, 'w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    # 保存配置信息
    config = {
        "num_classes": len(classes),
        "classes": classes,
        "label_map": label_map,
        "dataset_root": dataset_root
    }
    
    config_file = os.path.join(output_dir, "dataset_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ label_list.txt: {len(classes)} 个类别")
    print(f"  ✅ dataset_config.json: 配置信息")
    
    return classes, label_map

def main():
    """
    主函数 - 完整的数据预处理流程
    """
    print("🐟 海洋鱼类数据集预处理系统")
    print("=" * 60)
    
    # 路径配置  
    original_dataset = "../../dataset"                       # 原始数据集
    balanced_dataset = "../../dataset_balanced"              # 均衡后数据集  
    final_dataset = "../../dataset_processed"               # 最终处理后的数据集
    config_output = "./"                                     # 配置文件输出目录
    
    # 检查原始数据集
    if not os.path.exists(original_dataset):
        print(f"❌ 原始数据集不存在: {original_dataset}")
        return
    
    try:
        # 步骤1: 分析原始数据集
        print("\n" + "="*60)
        classes, class_stats = analyze_dataset(original_dataset)
        
        # 步骤2: 数据均衡化
        print("\n" + "="*60)
        balanced_stats = balance_dataset(
            original_dataset, 
            balanced_dataset,
            max_samples_per_class=200,  # 每类最多200张
            min_samples_per_class=20    # 每类至少20张
        )
        
        # 步骤3: 划分数据集
        print("\n" + "="*60)
        split_stats = split_balanced_dataset(
            balanced_dataset,
            final_dataset,
            train_ratio=0.7,    # 70%训练
            val_ratio=0.15,     # 15%验证  
            test_ratio=0.15     # 15%测试
        )
        
        # 步骤4: 生成PaddleX格式文件
        print("\n" + "="*60)
        classes, label_map = generate_paddlex_format(final_dataset, config_output)
        
        # 总结
        print("\n" + "="*60)
        print("🎉 数据预处理完成!")
        print(f"\n📁 生成的文件和目录:")
        print(f"  原始数据集:     {original_dataset}")
        print(f"  均衡数据集:     {balanced_dataset}")
        print(f"  最终数据集:     {final_dataset}")
        print(f"    ├── train/   (训练集)")
        print(f"    ├── val/     (验证集)")
        print(f"    └── test/    (测试集)")
        print(f"\n📝 训练配置文件:")
        print(f"  ├── train_list.txt      (训练数据列表)")
        print(f"  ├── val_list.txt        (验证数据列表)")
        print(f"  ├── test_list.txt       (测试数据列表)")
        print(f"  ├── label_list.txt      (类别标签)")
        print(f"  └── dataset_config.json (数据集配置)")
        
        print(f"\n📊 最终统计:")
        train_total = sum(split_stats['train'].values())
        val_total = sum(split_stats['val'].values())
        test_total = sum(split_stats['test'].values())
        total = train_total + val_total + test_total
        
        print(f"  类别数量: {len(classes)}")
        print(f"  训练样本: {train_total:,} 张 ({train_total/total:.1%})")
        print(f"  验证样本: {val_total:,} 张 ({val_total/total:.1%})")
        print(f"  测试样本: {test_total:,} 张 ({test_total/total:.1%})")
        print(f"  总样本数: {total:,} 张")
        
        print(f"\n🚀 下一步:")
        print("  1. 检查生成的数据集结构")
        print("  2. 运行 'python train_resnet_model.py' 开始训练")
        print("  3. 使用ResNet50 + 迁移学习训练鱼类分类模型")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    random.seed(42)
    main()
