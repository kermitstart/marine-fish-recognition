# 使用新版本PaddleX重新训练鱼类分类模型

import paddlex as pdx
import os
import json

def create_dataset_config():
    """
    创建PaddleX 3.x兼容的数据集配置
    """
    config = {
        "Global": {
            "dataset_dir": "../../../dataset",
            "device": "gpu"  # 如果有GPU的话，否则改为"cpu"
        },
        "Train": {
            "dataset": {
                "name": "ClsDataset", 
                "data_root": "../../../dataset",
                "train_list": "train_list.txt"
            },
            "dataloader": {
                "batch_size": 16,
                "num_workers": 4,
                "shuffle": True
            }
        },
        "Eval": {
            "dataset": {
                "name": "ClsDataset",
                "data_root": "../../../dataset", 
                "val_list": "val_list.txt"
            },
            "dataloader": {
                "batch_size": 16,
                "num_workers": 4,
                "shuffle": False
            }
        }
    }
    
    # 保存配置文件
    with open("dataset_config.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config

def train_fish_classification_model():
    """
    使用新版本PaddleX训练鱼类分类模型
    """
    
    print("🔄 准备数据集...")
    
    # 1. 准备数据集配置
    dataset_config = create_dataset_config()
    
    try:
        # 2. 创建分类管道 - 使用PaddleX 3.x API
        print("🤖 创建分类管道...")
        
        # 尝试使用图像分类管道
        pipeline = pdx.create_pipeline(
            pipeline="image_classification",
            llm_name=None,  # 不使用大语言模型
            device="gpu"    # 使用CPU，如果有GPU可以改为"gpu"
        )
        
        # 3. 训练模型
        print("🚀 开始训练模型...")
        
        # 创建训练配置
        train_config = {
            "epochs": 30,           # 训练轮数
            "learning_rate": 0.001, # 学习率  
            "batch_size": 16,       # 批次大小
            "save_dir": "./output_new"  # 保存路径
        }
        
        # 开始训练
        pipeline.train(
            train_dataset="train_list.txt",
            eval_dataset="val_list.txt", 
            **train_config
        )
        
        # 4. 评估模型
        print("📊 评估模型性能...")
        eval_result = pipeline.evaluate("val_list.txt")
        print(f"评估结果: {eval_result}")
        
        # 5. 导出模型
        print("💾 导出模型...")
        pipeline.export(save_dir="./inference_model_new")
        
        print("✅ 模型训练完成！")
        return True
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        print("尝试使用替代训练方法...")
        return train_with_alternative_method()

def train_with_alternative_method():
    """
    使用替代方法训练模型
    """
    try:
        print("� 使用简化训练方法...")
        
        # 创建简单的图像分类器
        model = pdx.create_model("PP-LCNet_x1_0")
        
        # 简化的训练配置
        train_result = model.train(
            train_list="train_list.txt",
            eval_list="val_list.txt",
            num_classes=23,  # 23种鱼类
            epochs=20,
            learning_rate=0.001,
            batch_size=8,   # 降低批次大小以减少内存使用
            save_dir="./output_simple"
        )
        
        print("� 训练结果:", train_result)
        
        # 导出模型
        model.export(save_dir="./inference_model_simple")
        
        print("✅ 简化训练完成！")
        return True
        
    except Exception as e:
        print(f"❌ 简化训练也失败了: {e}")
        print("建议检查PaddleX版本和数据集格式")
        return False

def prepare_dataset_files():
    """
    准备数据集配置文件 - 适配PaddleX格式
    """
    import glob
    import random
    import shutil
    
    dataset_root = "../../../dataset"
    
    print("🔍 分析数据集...")
    
    # 获取所有类别
    classes = [d for d in os.listdir(dataset_root) 
               if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')]
    classes.sort()
    
    print(f"发现 {len(classes)} 个鱼类类别:")
    total_images = 0
    class_stats = {}
    
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(dataset_root, cls)
        images = glob.glob(os.path.join(cls_dir, "*.png")) + \
                glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                glob.glob(os.path.join(cls_dir, "*.jpeg"))
        
        class_stats[cls] = len(images)
        total_images += len(images)
        print(f"  {i:2d}: {cls:<30} - {len(images):4d} 张图片")
    
    print(f"\n📊 数据集统计:")
    print(f"  总类别数: {len(classes)}")
    print(f"  总图片数: {total_images}")
    print(f"  平均每类: {total_images // len(classes)} 张")
    
    # 创建标签映射文件
    label_map = {}
    with open("label_list.txt", "w", encoding="utf-8") as f:
        for i, cls in enumerate(classes):
            f.write(f"{cls}\n")
            label_map[cls] = i
    
    # 准备训练和验证数据列表
    train_list = []
    val_list = []
    
    print("\n🔄 准备训练/验证数据分割...")
    
    for cls in classes:
        cls_dir = os.path.join(dataset_root, cls)
        images = glob.glob(os.path.join(cls_dir, "*.png")) + \
                glob.glob(os.path.join(cls_dir, "*.jpg")) + \
                glob.glob(os.path.join(cls_dir, "*.jpeg"))
        
        if len(images) == 0:
            print(f"⚠️  警告: {cls} 类别没有找到图片")
            continue
        
        # 随机打散
        random.shuffle(images)
        
        # 确保每个类别至少有1张验证图片
        if len(images) < 2:
            print(f"⚠️  警告: {cls} 类别图片太少({len(images)}张)，全部用作训练")
            split_idx = len(images)
        else:
            # 80%用于训练，20%用于验证，但至少保留1张用于验证
            split_idx = max(1, int(len(images) * 0.8))
            if split_idx == len(images):
                split_idx = len(images) - 1
        
        # 添加训练样本
        for img in images[:split_idx]:
            rel_path = os.path.relpath(img, dataset_root)
            train_list.append(f"{rel_path}\t{label_map[cls]}")
        
        # 添加验证样本
        for img in images[split_idx:]:
            rel_path = os.path.relpath(img, dataset_root)
            val_list.append(f"{rel_path}\t{label_map[cls]}")
    
    # 打散训练和验证列表
    random.shuffle(train_list)
    random.shuffle(val_list)
    
    # 保存列表文件
    with open("train_list.txt", "w", encoding="utf-8") as f:
        for item in train_list:
            f.write(f"{item}\n")
    
    with open("val_list.txt", "w", encoding="utf-8") as f:
        for item in val_list:
            f.write(f"{item}\n")
    
    # 创建类别名称映射文件
    with open("class_names.txt", "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    print(f"\n✅ 数据集准备完成:")
    print(f"  训练样本: {len(train_list)} 张")
    print(f"  验证样本: {len(val_list)} 张") 
    print(f"  训练/验证比例: {len(train_list)/(len(train_list)+len(val_list)):.1%} / {len(val_list)/(len(train_list)+len(val_list)):.1%}")
    
    # 检查数据平衡性
    print(f"\n📈 数据平衡性分析:")
    min_count = min(class_stats.values())
    max_count = max(class_stats.values())
    print(f"  最少: {min_count} 张")
    print(f"  最多: {max_count} 张") 
    print(f"  平衡比: {min_count/max_count:.2f}")
    
    if min_count / max_count < 0.1:
        print("  ⚠️  数据不平衡严重，建议进行数据增强")
    
    return classes, len(train_list), len(val_list)

def check_environment():
    """
    检查训练环境
    """
    print("🔧 检查训练环境...")
    
    # 检查PaddleX版本
    try:
        import paddlex as pdx
        print(f"✅ PaddleX版本: {pdx.__version__}")
    except Exception as e:
        print(f"❌ PaddleX导入失败: {e}")
        return False
    
    # 检查数据集路径
    dataset_path = "../../../dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    else:
        print(f"✅ 数据集路径: {dataset_path}")
    
    # 检查可用内存和GPU
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ 可用内存: {memory.available // (1024**3):.1f} GB / {memory.total // (1024**3):.1f} GB")
    except:
        print("⚠️  无法检查内存信息")
    
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            print("✅ GPU支持: 已启用")
        else:
            print("⚠️  GPU支持: 未启用，将使用CPU训练")
    except:
        print("⚠️  无法检查GPU支持")
    
    return True

def create_quick_test():
    """
    创建快速测试脚本
    """
    test_code = '''
# 快速测试训练好的模型
import paddlex as pdx
import os

def test_model():
    model_path = "./inference_model_new"  # 或 "./inference_model_simple" 
    
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在")
        return
    
    try:
        # 加载模型
        predictor = pdx.create_predictor(model_path)
        
        # 测试图片路径
        test_image = "../../../dataset/Abudefduf_vaigiensis/fish_000001719594_03397.png"
        
        if os.path.exists(test_image):
            result = predictor.predict([test_image])
            print("预测结果:", result)
        else:
            print("测试图片不存在")
            
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_model()
'''
    
    with open("test_model.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    
    print("✅ 已创建测试脚本: test_model.py")

if __name__ == "__main__":
    print("🐟 海洋鱼类识别模型训练系统")
    print("=" * 50)
    
    # 1. 检查环境
    if not check_environment():
        print("❌ 环境检查失败，请解决上述问题后重试")
        exit(1)
    
    print("\n" + "=" * 50)
    print("🔄 开始准备数据集...")
    
    # 2. 准备数据集文件  
    try:
        classes, train_count, val_count = prepare_dataset_files()
        
        if train_count == 0:
            print("❌ 没有找到训练数据")
            exit(1)
            
    except Exception as e:
        print(f"❌ 数据集准备失败: {e}")
        exit(1)
    
    print("\n" + "=" * 50)
    print("🚀 开始模型训练...")
    
    # 3. 训练模型
    try:
        success = train_fish_classification_model()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 训练完成！")
            
            # 创建测试脚本
            create_quick_test()
            
            print("\n📋 下一步:")
            print("1. 运行 'python test_model.py' 测试模型")
            print("2. 将训练好的模型替换到 './core/inference_model_new'")
            print("3. 更新后端代码使用新模型")
            
        else:
            print("❌ 训练失败，请检查错误信息")
            
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        print("建议:")
        print("- 检查数据集格式")
        print("- 降低batch_size")
        print("- 确保有足够的内存和存储空间")
