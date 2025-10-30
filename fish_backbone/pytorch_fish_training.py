# PyTorch版本的鱼类分类训练脚本
# 适用于Colab训练，使用ResNet50迁移学习

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import json
import glob
import random
import shutil
import time
from sklearn.metrics import accuracy_score, classification_report

class FishDataset(Dataset):
    """
    鱼类数据集类
    """
    def __init__(self, data_list, root_dir, transform=None):
        self.data_list = data_list
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img_full_path = os.path.join(self.root_dir, img_path)
        
        # 加载图片
        image = Image.open(img_full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def prepare_balanced_dataset(source_dir="../../dataset", 
                           target_dir="./balanced_dataset", 
                           max_samples_per_class=200,
                           test_split=0.2,
                           val_split=0.2):
    """
    准备平衡的数据集
    """
    print("🔄 准备平衡数据集...")
    
    # 清理目标目录
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # 创建目录
    train_dir = os.path.join(target_dir, "train")
    val_dir = os.path.join(target_dir, "val")
    test_dir = os.path.join(target_dir, "test")
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
    
    # 获取所有类别
    if not os.path.exists(source_dir):
        print(f"❌ 数据集目录不存在: {source_dir}")
        return None
    
    all_classes = [d for d in os.listdir(source_dir) 
                   if os.path.isdir(os.path.join(source_dir, d)) and not d.startswith('.')]
    all_classes.sort()
    
    print(f"发现 {len(all_classes)} 个类别")
    
    # 过滤掉样本太少的类别
    valid_classes = []
    class_stats = {}
    
    for class_name in all_classes:
        class_dir = os.path.join(source_dir, class_name)
        images = glob.glob(os.path.join(class_dir, "*.png")) + \
                glob.glob(os.path.join(class_dir, "*.jpg")) + \
                glob.glob(os.path.join(class_dir, "*.jpeg"))
        
        # 至少需要10张图片
        if len(images) >= 10:
            valid_classes.append(class_name)
            class_stats[class_name] = len(images)
            print(f"  {class_name}: {len(images)} 张")
        else:
            print(f"  {class_name}: {len(images)} 张 (跳过-样本太少)")
    
    print(f"\\n有效类别: {len(valid_classes)}")
    
    # 处理每个类别
    train_data = []
    val_data = []
    test_data = []
    
    for i, class_name in enumerate(valid_classes):
        class_dir = os.path.join(source_dir, class_name)
        images = glob.glob(os.path.join(class_dir, "*.png")) + \
                glob.glob(os.path.join(class_dir, "*.jpg")) + \
                glob.glob(os.path.join(class_dir, "*.jpeg"))
        
        random.shuffle(images)
        
        # 限制每类最大样本数
        if len(images) > max_samples_per_class:
            images = images[:max_samples_per_class]
        
        # 划分数据集
        n_test = max(1, int(len(images) * test_split))
        n_val = max(1, int(len(images) * val_split))
        n_train = len(images) - n_test - n_val
        
        test_images = images[:n_test]
        val_images = images[n_test:n_test+n_val]
        train_images = images[n_test+n_val:]
        
        # 创建类别目录
        for split, split_dir in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
            class_split_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_split_dir, exist_ok=True)
        
        # 复制文件并记录
        for img_path in train_images:
            filename = os.path.basename(img_path)
            target_path = os.path.join(train_dir, class_name, filename)
            shutil.copy2(img_path, target_path)
            rel_path = os.path.join("train", class_name, filename)
            train_data.append((rel_path, i))
        
        for img_path in val_images:
            filename = os.path.basename(img_path)
            target_path = os.path.join(val_dir, class_name, filename)
            shutil.copy2(img_path, target_path)
            rel_path = os.path.join("val", class_name, filename)
            val_data.append((rel_path, i))
            
        for img_path in test_images:
            filename = os.path.basename(img_path)
            target_path = os.path.join(test_dir, class_name, filename)
            shutil.copy2(img_path, target_path)
            rel_path = os.path.join("test", class_name, filename)
            test_data.append((rel_path, i))
    
    # 保存配置
    config = {
        "dataset_root": target_dir,
        "num_classes": len(valid_classes),
        "classes": valid_classes,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "class_stats": class_stats
    }
    
    with open("pytorch_dataset_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\\n✅ 数据集准备完成:")
    print(f"  类别数: {len(valid_classes)}")
    print(f"  训练样本: {len(train_data)}")
    print(f"  验证样本: {len(val_data)}")
    print(f"  测试样本: {len(test_data)}")
    
    return config, train_data, val_data, test_data

def create_model(num_classes, pretrained=True):
    """
    创建ResNet50模型
    """
    model = models.resnet50(pretrained=pretrained)
    
    # 替换最后的分类层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, config, 
                epochs=30, lr=0.001, device='cpu'):
    """
    训练模型
    """
    print(f"🚀 开始训练 (设备: {device})...")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_acc = 0.0
    train_history = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # 计算准确率
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"时间: {epoch_time:.1f}s | "
              f"训练损失: {train_loss/len(train_loader):.4f} | "
              f"训练准确率: {train_acc:.2f}% | "
              f"验证损失: {val_loss/len(val_loader):.4f} | "
              f"验证准确率: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, 'best_fish_model.pth')
        
        # 记录训练历史
        train_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': train_acc,
            'val_loss': val_loss / len(val_loader),
            'val_acc': val_acc
        })
        
        scheduler.step()
    
    print(f"\\n✅ 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    return train_history

def main():
    """
    主训练流程
    """
    print("🐟 PyTorch 鱼类分类训练系统")
    print("=" * 50)
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    print("\\n准备数据集...")
    config, train_data, val_data, test_data = prepare_balanced_dataset()
    
    if config is None:
        return
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和加载器
    train_dataset = FishDataset(train_data, config['dataset_root'], train_transform)
    val_dataset = FishDataset(val_data, config['dataset_root'], val_transform)
    
    batch_size = 16 if device.type == 'cuda' else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 创建模型
    print(f"\\n创建ResNet50模型 ({config['num_classes']} 个类别)...")
    model = create_model(config['num_classes'])
    
    # 训练模型
    train_history = train_model(
        model, train_loader, val_loader, config,
        epochs=20,  # 可以在Colab中调整为更多轮数
        lr=0.001,
        device=device
    )
    
    print("\\n🎉 训练完成!")
    print("\\n📋 后续步骤:")
    print("1. 检查 'best_fish_model.pth' 文件")
    print("2. 可以加载模型进行推理测试")
    print("3. 转换为PaddlePaddle格式(可选)")

if __name__ == "__main__":
    main()
