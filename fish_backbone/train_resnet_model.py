# ResNet50迁移学习训练脚本 - 针对海洋鱼类识别
# 使用预训练的ResNet50模型进行微调

import paddlex as pdx
import paddle
import os
import json
import time
import numpy as np

class FishClassificationTrainer:
    """
    鱼类分类训练器 - 使用ResNet50迁移学习
    """
    
    def __init__(self, config_path="./dataset_config.json"):
        """
        初始化训练器
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.predictor = None
        
    def load_config(self):
        """
        加载数据集配置
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 加载数据集配置: {config['num_classes']} 个类别")
            return config
        else:
            print(f"❌ 配置文件不存在: {self.config_path}")
            print("请先运行 data_preprocess.py 处理数据集")
            return None
    
    def setup_training_environment(self):
        """
        设置训练环境
        """
        print("🔧 设置训练环境...")
        
        # 检查设备
        if paddle.is_compiled_with_cuda() and paddle.device.get_device():
            device = "gpu"
            print("✅ 使用GPU训练")
        else:
            device = "cpu" 
            print("⚠️  使用CPU训练（建议使用GPU以获得更好性能）")
        
        # 检查内存
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            print(f"💾 可用内存: {available_gb:.1f} GB")
            
            if available_gb < 4:
                print("⚠️  内存较少，建议降低batch_size")
        except:
            pass
            
        return device
    
    def create_model_pipeline(self, device="cpu"):
        """
        创建基于ResNet50的图像分类管道
        """
        print("🤖 创建ResNet50分类管道...")
        
        try:
            # 方法1: 使用PaddleX的图像分类管道
            pipeline = pdx.create_pipeline(
                pipeline="image_classification",
                device=device
            )
            print("✅ 成功创建分类管道")
            return pipeline, "pipeline"
            
        except Exception as e1:
            print(f"⚠️  管道创建失败: {e1}")
            
            try:
                # 方法2: 直接使用预训练模型
                print("🔄 尝试使用预训练ResNet50模型...")
                model = pdx.create_model("ResNet50")
                print("✅ 成功创建ResNet50模型")
                return model, "model"
                
            except Exception as e2:
                print(f"⚠️  ResNet50创建失败: {e2}")
                
                try:
                    # 方法3: 使用轻量级模型
                    print("🔄 尝试使用PP-LCNet轻量级模型...")
                    model = pdx.create_model("PP-LCNet_x1_0")
                    print("✅ 成功创建PP-LCNet模型")
                    return model, "lightweight"
                    
                except Exception as e3:
                    print(f"❌ 所有模型创建方法都失败了:")
                    print(f"  Pipeline: {e1}")
                    print(f"  ResNet50: {e2}")
                    print(f"  PP-LCNet: {e3}")
                    return None, None
    
    def train_model(self, epochs=50, batch_size=16, learning_rate=0.001):
        """
        训练模型 - 使用迁移学习策略
        """
        print("🚀 开始迁移学习训练...")
        
        # 设置环境
        device = self.setup_training_environment()
        
        # 创建模型
        model, model_type = self.create_model_pipeline(device)
        if model is None:
            return False
        
        # 调整参数（根据可用资源）
        if device == "cpu":
            batch_size = max(4, batch_size // 4)  # CPU时减小batch_size
            print(f"🔧 调整CPU训练参数: batch_size={batch_size}")
        
        # 训练配置
        train_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "save_dir": "./output_resnet50",
            "save_interval_epochs": 5,  # 每5个epoch保存一次
            "log_interval_steps": 10,   # 每10步输出一次日志
        }
        
        print(f"\n📋 训练配置:")
        print(f"  模型类型: {model_type}")
        print(f"  训练轮数: {epochs}")
        print(f"  批次大小: {batch_size}")
        print(f"  学习率:   {learning_rate}")
        print(f"  设备:     {device}")
        print(f"  类别数:   {self.config['num_classes']}")
        
        try:
            # 开始训练
            start_time = time.time()
            
            if model_type == "pipeline":
                # 使用管道训练
                result = model.train(
                    train_dataset="train_list.txt",
                    eval_dataset="val_list.txt",
                    **train_config
                )
            else:
                # 使用模型训练
                result = model.train(
                    train_list="train_list.txt",
                    eval_list="val_list.txt", 
                    num_classes=self.config['num_classes'],
                    **train_config
                )
            
            training_time = time.time() - start_time
            
            print(f"\n✅ 训练完成!")
            print(f"  训练时间: {training_time/3600:.2f} 小时")
            print(f"  训练结果: {result}")
            
            # 保存模型
            self.model = model
            return True
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def evaluate_model(self):
        """
        评估模型性能
        """
        if self.model is None:
            print("❌ 没有可用的模型进行评估")
            return None
        
        print("📊 评估模型性能...")
        
        try:
            # 在验证集上评估
            eval_result = self.model.evaluate("val_list.txt")
            print(f"验证集评估结果: {eval_result}")
            
            # 在测试集上评估（如果存在）
            if os.path.exists("test_list.txt"):
                test_result = self.model.evaluate("test_list.txt")
                print(f"测试集评估结果: {test_result}")
            
            return eval_result
            
        except Exception as e:
            print(f"⚠️  评估失败: {e}")
            return None
    
    def export_model(self, export_dir="./inference_model_resnet50"):
        """
        导出训练好的模型用于推理
        """
        if self.model is None:
            print("❌ 没有可用的模型进行导出")
            return False
        
        print(f"💾 导出模型到: {export_dir}")
        
        try:
            self.model.export(save_dir=export_dir)
            print("✅ 模型导出成功!")
            
            # 验证导出的模型
            if os.path.exists(export_dir):
                files = os.listdir(export_dir)
                print(f"导出文件: {files}")
                return True
            else:
                print("❌ 导出目录不存在")
                return False
                
        except Exception as e:
            print(f"❌ 模型导出失败: {e}")
            return False
    
    def test_exported_model(self, model_dir="./inference_model_resnet50"):
        """
        测试导出的模型
        """
        if not os.path.exists(model_dir):
            print(f"❌ 模型目录不存在: {model_dir}")
            return False
        
        print(f"🧪 测试导出的模型...")
        
        try:
            # 加载导出的模型
            predictor = pdx.create_predictor(model_dir)
            self.predictor = predictor
            
            # 找一张测试图片
            test_image = None
            dataset_root = self.config.get('dataset_root', '../../dataset_processed')
            
            # 尝试从测试集找图片
            for split in ['test', 'val', 'train']:
                split_dir = os.path.join(dataset_root, split)
                if os.path.exists(split_dir):
                    for cls in os.listdir(split_dir):
                        cls_dir = os.path.join(split_dir, cls)
                        if os.path.isdir(cls_dir):
                            images = [f for f in os.listdir(cls_dir) 
                                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                            if images:
                                test_image = os.path.join(cls_dir, images[0])
                                expected_class = cls
                                break
                    if test_image:
                        break
            
            if test_image and os.path.exists(test_image):
                print(f"  测试图片: {test_image}")
                print(f"  期望类别: {expected_class}")
                
                result = predictor.predict([test_image])
                print(f"  预测结果: {result}")
                
                # 解析预测结果
                if result and len(result) > 0:
                    pred = result[0]
                    if 'class_ids' in pred and 'scores' in pred:
                        class_id = pred['class_ids'][0]
                        score = pred['scores'][0]
                        predicted_class = self.config['classes'][class_id]
                        
                        print(f"  预测类别: {predicted_class}")
                        print(f"  置信度:   {score:.4f}")
                        
                        if predicted_class == expected_class:
                            print("  ✅ 预测正确!")
                        else:
                            print("  ❌ 预测错误")
                
                return True
            else:
                print("  ⚠️  找不到测试图片")
                return False
                
        except Exception as e:
            print(f"❌ 模型测试失败: {e}")
            return False

def main():
    """
    主训练流程
    """
    print("🐟 ResNet50海洋鱼类分类训练系统")
    print("=" * 60)
    
    # 检查数据预处理是否完成
    if not os.path.exists("dataset_config.json"):
        print("❌ 请先运行 'python data_preprocess.py' 处理数据集")
        return
    
    if not os.path.exists("train_list.txt"):
        print("❌ 缺少训练数据列表文件，请先运行数据预处理")
        return
    
    # 创建训练器
    trainer = FishClassificationTrainer()
    
    if trainer.config is None:
        print("❌ 配置加载失败")
        return
    
    try:
        print("\n" + "="*60)
        print("🚀 开始ResNet50迁移学习训练...")
        
        # 训练模型
        success = trainer.train_model(
            epochs=30,              # 训练轮数
            batch_size=16,          # 批次大小
            learning_rate=0.001     # 学习率
        )
        
        if not success:
            print("❌ 训练失败")
            return
        
        print("\n" + "="*60)
        # 评估模型
        trainer.evaluate_model()
        
        print("\n" + "="*60)
        # 导出模型
        export_success = trainer.export_model()
        
        if export_success:
            print("\n" + "="*60)
            # 测试导出的模型
            trainer.test_exported_model()
            
            print("\n" + "="*60)
            print("🎉 训练流程完成!")
            print("\n📋 下一步:")
            print("1. 检查 './output_resnet50' 目录中的训练日志")
            print("2. 使用 './inference_model_resnet50' 目录中的模型进行推理")
            print("3. 将模型集成到 app_updated.py 中")
            
            print("\n🔧 集成到后端系统:")
            print("1. 复制模型文件:")
            print("   cp -r ./inference_model_resnet50 ./core/inference_model_new")
            print("2. 更新后端代码使用新模型")
            print("3. 重启后端服务测试效果")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
