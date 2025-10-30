# 简化测试 - 验证PaddleX训练API

import paddlex as pdx
import os
import json

def test_paddlex_apis():
    """
    测试PaddleX的不同训练API
    """
    print("🧪 测试PaddleX训练API...")
    
    try:
        # 方法1: 尝试build_trainer
        print("\n1. 测试 build_trainer API...")
        trainer = pdx.build_trainer(
            task="image_classification",
            model="PP-LCNet_x0_5"
        )
        print(f"✅ Trainer创建成功: {type(trainer)}")
        print(f"可用方法: {[m for m in dir(trainer) if not m.startswith('_')]}")
        
        return True
        
    except Exception as e1:
        print(f"❌ build_trainer失败: {e1}")
        
        try:
            # 方法2: 检查推理管道功能
            print("\n2. 测试推理管道...")
            pipeline = pdx.create_pipeline("image_classification")
            print(f"✅ Pipeline创建成功: {type(pipeline)}")
            print(f"可用方法: {[m for m in dir(pipeline) if not m.startswith('_') and 'train' in m.lower()]}")
            
            return False  # 只能推理，不能训练
            
        except Exception as e2:
            print(f"❌ Pipeline失败: {e2}")
            return False

def test_simple_inference():
    """
    测试简单推理功能
    """
    print("\n🔍 测试简单推理功能...")
    
    try:
        # 创建推理管道
        pipeline = pdx.create_pipeline("image_classification")
        
        # 找一张测试图片
        test_image = None
        for root, dirs, files in os.walk("../../dataset"):
            for file in files:
                if file.lower().endswith(('.png', '.jpg')):
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break
        
        if test_image:
            print(f"  测试图片: {test_image}")
            result = pipeline.predict([test_image])
            print(f"  预测结果: {result}")
            print("✅ 推理测试成功")
            return True
        else:
            print("❌ 找不到测试图片")
            return False
            
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        return False

if __name__ == "__main__":
    print("🧪 PaddleX API 测试")
    print("=" * 40)
    
    # 测试训练API
    train_available = test_paddlex_apis()
    
    # 测试推理API
    inference_available = test_simple_inference()
    
    print("\n" + "=" * 40)
    print("📋 测试结果:")
    print(f"  训练功能: {'✅ 可用' if train_available else '❌ 不可用'}")
    print(f"  推理功能: {'✅ 可用' if inference_available else '❌ 不可用'}")
    
    if not train_available:
        print("\n💡 建议:")
        print("1. PaddleX 3.x 主要用于推理")
        print("2. 训练可能需要使用 PaddlePaddle 原生API")
        print("3. 或者使用 PaddleClas 进行图像分类训练")
        print("4. 考虑使用 PyTorch 或 TensorFlow 进行训练")
