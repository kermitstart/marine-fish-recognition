# 快速测试训练脚本 - 验证代码可行性
# 使用小数据集和少量epoch进行测试

import paddlex as pdx
import paddle
import os
import json
import time
import glob
import random
import shutil

def create_mini_dataset(source_dir="../../dataset", target_dir="./mini_dataset", samples_per_class=10):
    """
    创建小型测试数据集
    """
    print(f"🔄 创建小型测试数据集...")
    print(f"  源目录: {source_dir}")
    print(f"  目标目录: {target_dir}")
    print(f"  每类样本数: {samples_per_class}")
    
    # 清理目标目录
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # 创建目录结构
    train_dir = os.path.join(target_dir, "train")
    val_dir = os.path.join(target_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # 获取前5个类别进行测试
    if not os.path.exists(source_dir):
        print(f"❌ 数据集目录不存在: {source_dir}")
        return None, None
    
    all_classes = [d for d in os.listdir(source_dir) 
                   if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith('.')]
    all_classes.sort()
    
    # 选择前5个类别进行快速测试
    test_classes = all_classes[:5]
    print(f"  测试类别: {test_classes}")
    
    train_files = []
    val_files = []
    class_stats = {}
    
    for i, class_name in enumerate(test_classes):
        source_class_dir = os.path.join(source_dir, class_name)
        
        # 获取该类别的所有图片
        images = glob.glob(os.path.join(source_class_dir, "*.png")) + \
                glob.glob(os.path.join(source_class_dir, "*.jpg")) + \
                glob.glob(os.path.join(source_class_dir, "*.jpeg"))
        
        if len(images) == 0:
            print(f"  ⚠️  {class_name} 类别没有图片")
            continue
        
        # 随机选择样本
        random.shuffle(images)
        selected_images = images[:samples_per_class]
        
        # 80%训练，20%验证
        split_idx = max(1, int(len(selected_images) * 0.8))
        train_images = selected_images[:split_idx]
        val_images = selected_images[split_idx:]
        
        # 创建类别目录
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # 复制训练图片
        for img_path in train_images:
            filename = os.path.basename(img_path)
            target_path = os.path.join(train_class_dir, filename)
            shutil.copy2(img_path, target_path)
            
            rel_path = os.path.join("train", class_name, filename)
            train_files.append(f"{rel_path}\t{i}")
        
        # 复制验证图片
        for img_path in val_images:
            filename = os.path.basename(img_path)
            target_path = os.path.join(val_class_dir, filename)
            shutil.copy2(img_path, target_path)
            
            rel_path = os.path.join("val", class_name, filename)
            val_files.append(f"{rel_path}\t{i}")
        
        class_stats[class_name] = {
            'train': len(train_images),
            'val': len(val_images),
            'total': len(selected_images)
        }
        
        print(f"    {class_name}: 训练{len(train_images)}张, 验证{len(val_images)}张")
    
    # 保存数据列表文件
    with open("mini_train_list.txt", "w", encoding="utf-8") as f:
        for item in train_files:
            f.write(f"{item}\n")
    
    with open("mini_val_list.txt", "w", encoding="utf-8") as f:
        for item in val_files:
            f.write(f"{item}\n")
    
    # 保存类别列表
    with open("mini_class_names.txt", "w", encoding="utf-8") as f:
        for cls in test_classes:
            f.write(f"{cls}\n")
    
    # 保存配置
    config = {
        "dataset_root": target_dir,
        "num_classes": len(test_classes),
        "classes": test_classes,
        "train_samples": len(train_files),
        "val_samples": len(val_files),
        "class_stats": class_stats
    }
    
    with open("mini_dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 小型数据集创建完成:")
    print(f"  类别数: {len(test_classes)}")
    print(f"  训练样本: {len(train_files)}")
    print(f"  验证样本: {len(val_files)}")
    
    return test_classes, config

def quick_test_training():
    """
    快速测试训练
    """
    print("🧪 开始快速测试训练...")
    
    # 检查环境
    print(f"  PaddleX版本: {pdx.__version__}")
    print(f"  Paddle版本: {paddle.__version__}")
    
    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    print(f"  设备: {device}")
    
    try:
        # 创建图像分类管道进行测试
        print("🤖 创建分类管道...")
        pipeline = pdx.create_pipeline(
            pipeline="image_classification",
            device=device
        )
        print("✅ 分类管道创建成功")
        
        # 训练配置 - 极小规模测试
        train_config = {
            "epochs": 3,       # 只训练3个epoch
            "batch_size": 4,   # 小批次
            "learning_rate": 0.001,
            "save_dir": "./test_output"
        }
        
        print(f"\n📋 测试训练配置:")
        for key, value in train_config.items():
            print(f"  {key}: {value}")
        
        # 开始训练
        print(f"\n🚀 开始测试训练（预计1-3分钟）...")
        start_time = time.time()
        
        result = pipeline.train(
            train_dataset="mini_train_list.txt",
            eval_dataset="mini_val_list.txt",
            **train_config
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ 测试训练完成!")
        print(f"  训练时间: {training_time:.1f} 秒")
        print(f"  训练结果: {result}")
        
        # 测试推理
        print(f"\n🔍 测试模型推理...")
        
        # 找一张测试图片
        test_image = None
        for root, dirs, files in os.walk("./mini_dataset/val"):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break
        
        if test_image:
            print(f"  测试图片: {test_image}")
            pred_result = pipeline.predict([test_image])
            print(f"  预测结果: {pred_result}")
        
        # 导出模型
        print(f"\n💾 导出测试模型...")
        pipeline.export(save_dir="./test_inference_model")
        print("✅ 模型导出成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    主测试流程
    """
    print("🧪 快速测试训练系统")
    print("=" * 50)
    
    try:
        # 1. 创建小型数据集
        test_classes, config = create_mini_dataset()
        
        if test_classes is None:
            print("❌ 小型数据集创建失败")
            return
        
        # 2. 进行快速训练测试
        print("\n" + "=" * 50)
        success = quick_test_training()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 快速测试成功!")
            print("\n📋 测试结果:")
            print("✅ 代码运行正常")
            print("✅ 模型训练成功") 
            print("✅ 模型推理正常")
            print("✅ 模型导出成功")
            
            print(f"\n🚀 可以进行完整训练了!")
            print(f"建议在Colab中:")
            print(f"1. 上传完整数据集")
            print(f"2. 使用GPU训练")
            print(f"3. 增加训练轮数到30-50")
            print(f"4. 使用完整的23个类别")
            
        else:
            print("\n❌ 快速测试失败")
            print("需要修复错误后再进行完整训练")
    
    except KeyboardInterrupt:
        print("\n⚠️  测试被中断")
    except Exception as e:
        print(f"\n❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
