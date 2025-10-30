# 测试训练好的PyTorch鱼类分类模型

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import glob
import random

def load_trained_model(model_path="best_fish_model.pth", config_path="pytorch_dataset_config.json"):
    """
    加载训练好的模型
    """
    print("🔄 加载训练好的模型...")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    num_classes = config['num_classes']
    classes = config['classes']
    
    # 重建模型结构
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"  类别数: {num_classes}")
    print(f"  验证准确率: {checkpoint['val_acc']:.2f}%")
    
    return model, classes, config

def preprocess_image(image_path):
    """
    预处理图像
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return image_tensor

def predict_single_image(model, image_path, classes):
    """
    预测单张图片
    """
    image_tensor = preprocess_image(image_path)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
        predicted_class = classes[predicted.item()]
        confidence_score = confidence.item()
    
    return predicted_class, confidence_score, probabilities

def test_random_images(model, classes, config, num_tests=10):
    """
    随机测试一些图片
    """
    print(f"\\n🧪 随机测试 {num_tests} 张图片...")
    
    dataset_root = config['dataset_root']
    correct_predictions = 0
    
    test_results = []
    
    for i in range(num_tests):
        # 随机选择一个类别
        random_class = random.choice(classes)
        class_dir = os.path.join(dataset_root, "test", random_class)
        
        # 如果测试集中没有这个类别，尝试验证集
        if not os.path.exists(class_dir):
            class_dir = os.path.join(dataset_root, "val", random_class)
        
        if not os.path.exists(class_dir):
            continue
            
        # 随机选择一张图片
        images = glob.glob(os.path.join(class_dir, "*.png")) + \
                glob.glob(os.path.join(class_dir, "*.jpg")) + \
                glob.glob(os.path.join(class_dir, "*.jpeg"))
        
        if not images:
            continue
            
        random_image = random.choice(images)
        
        # 预测
        predicted_class, confidence, probabilities = predict_single_image(model, random_image, classes)
        
        # 计算准确率
        is_correct = predicted_class == random_class
        if is_correct:
            correct_predictions += 1
        
        # 获取top-3预测
        top3_indices = torch.topk(probabilities, 3).indices
        top3_classes = [classes[idx] for idx in top3_indices]
        top3_scores = [probabilities[idx].item() for idx in top3_indices]
        
        result = {
            'image': os.path.basename(random_image),
            'true_class': random_class,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'correct': is_correct,
            'top3': list(zip(top3_classes, top3_scores))
        }
        
        test_results.append(result)
        
        status = "✅" if is_correct else "❌"
        print(f"  {i+1:2d}. {status} {os.path.basename(random_image)}")
        print(f"      真实: {random_class}")
        print(f"      预测: {predicted_class} ({confidence:.3f})")
        if not is_correct:
            print(f"      Top3: {', '.join([f'{cls}({score:.3f})' for cls, score in result['top3'][:3]])}")
        print()
    
    accuracy = correct_predictions / len(test_results) * 100 if test_results else 0
    print(f"\\n📊 测试结果:")
    print(f"  测试样本: {len(test_results)}")
    print(f"  正确预测: {correct_predictions}")
    print(f"  准确率: {accuracy:.2f}%")
    
    return test_results

def interactive_test():
    """
    交互式测试
    """
    print("\\n🎯 交互式测试")
    print("输入图片路径进行测试 (输入 'quit' 退出):")
    
    while True:
        image_path = input("\\n图片路径: ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            break
            
        if not os.path.exists(image_path):
            print("❌ 文件不存在")
            continue
        
        try:
            predicted_class, confidence, probabilities = predict_single_image(model, image_path, classes)
            
            print(f"\\n预测结果:")
            print(f"  类别: {predicted_class}")
            print(f"  置信度: {confidence:.3f}")
            
            # 显示top-5预测
            top5_indices = torch.topk(probabilities, 5).indices
            print(f"\\n  Top-5预测:")
            for i, idx in enumerate(top5_indices):
                class_name = classes[idx]
                score = probabilities[idx].item()
                print(f"    {i+1}. {class_name}: {score:.3f}")
                
        except Exception as e:
            print(f"❌ 预测失败: {e}")

def main():
    """
    主测试函数
    """
    print("🧪 PyTorch鱼类分类模型测试")
    print("=" * 50)
    
    # 检查文件
    if not os.path.exists("best_fish_model.pth"):
        print("❌ 找不到模型文件 'best_fish_model.pth'")
        print("请先运行训练脚本")
        return
    
    if not os.path.exists("pytorch_dataset_config.json"):
        print("❌ 找不到配置文件")
        return
    
    # 加载模型
    global model, classes, config
    model, classes, config = load_trained_model()
    
    print(f"\\n📋 支持的鱼类类别 ({len(classes)}种):")
    for i, cls in enumerate(classes):
        print(f"  {i+1:2d}. {cls}")
    
    # 随机测试
    test_results = test_random_images(model, classes, config, num_tests=10)
    
    # 交互式测试
    try:
        interactive_test()
    except KeyboardInterrupt:
        print("\\n退出测试")
    
    print("\\n🎉 测试完成!")
    print("\\n📋 模型已准备就绪:")
    print("1. 可以集成到Web后端")
    print("2. 可以转换为其他格式")
    print("3. 可以进一步优化")

if __name__ == "__main__":
    main()
