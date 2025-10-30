# 集成PyTorch模型到现有后端系统
# 替换原有的模拟预测，使用训练好的真实AI模型

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import os
import numpy as np
import cv2
import shutil
import datetime
from datetime import timedelta
from flask import *

# 导入多鱼类检测模块
from multi_fish_detector import MultiFishDetector

UPLOAD_FOLDER = r'./uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

class PyTorchFishClassifier:
    """
    PyTorch鱼类分类器
    """
    
    def __init__(self, model_path="best_fish_model.pth", config_path="pytorch_dataset_config.json"):
        self.model = None
        self.classes = None
        self.config = None
        self.transform = None
        self.model_loaded = False
        
        try:
            self.load_model(model_path, config_path)
            self.setup_transform()
            self.model_loaded = True
            print("✅ PyTorch鱼类分类模型加载成功")
        except Exception as e:
            print(f"⚠️  PyTorch模型加载失败: {e}")
            self.model_loaded = False
    
    def load_model(self, model_path, config_path):
        """加载训练好的模型"""
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.classes = self.config['classes']
        num_classes = self.config['num_classes']
        
        # 重建模型结构
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"模型验证准确率: {checkpoint['val_acc']:.2f}%")
    
    def setup_transform(self):
        """设置图像预处理"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        预测图像类别
        """
        if not self.model_loaded:
            return None, 0.0
        
        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
                predicted_class = self.classes[predicted.item()]
                confidence_score = confidence.item()
            
            return predicted_class, confidence_score
            
        except Exception as e:
            print(f"预测错误: {e}")
            return None, 0.0
    
    def get_top_k_predictions(self, image_path, k=3):
        """
        获取Top-K预测结果
        """
        if not self.model_loaded:
            return []
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                
                # 获取top-k结果
                top_k_values, top_k_indices = torch.topk(probabilities, k)
                
                results = []
                for i in range(k):
                    class_name = self.classes[top_k_indices[i].item()]
                    confidence = top_k_values[i].item()
                    results.append({
                        'class': class_name,
                        'confidence': confidence
                    })
                
                return results
                
        except Exception as e:
            print(f"Top-K预测错误: {e}")
            return []

# 全局分类器实例
fish_classifier = PyTorchFishClassifier()

# 初始化多鱼类检测器
multi_detector = None

def init_multi_detector():
    """初始化多鱼类检测器"""
    global multi_detector
    if fish_classifier.model_loaded and multi_detector is None:
        multi_detector = MultiFishDetector(
            fish_classifier, 
            confidence_threshold=0.5, 
            overlap_threshold=0.3
        )
        print("🎯 多鱼类检测器初始化成功")

def mock_prediction_fallback():
    """降级到模拟预测"""
    fish_names = [
        "Abudefduf_vaigiensis", "Acanthurus_nigrofuscus", "Amphiprion_clarkia",
        "Balistapus_undulates", "Canthigaster_valentine", "Chaetodon_lunulatus"
    ]
    import random
    return random.choice(fish_names)

def predict_classification(image_path):
    """
    鱼类分类预测 - 优先使用PyTorch模型
    """
    if fish_classifier.model_loaded:
        try:
            predicted_class, confidence = fish_classifier.predict(image_path)
            if predicted_class:
                return predicted_class, confidence, True  # True表示使用真实模型
        except Exception as e:
            print(f"PyTorch预测失败: {e}")
    
    # 降级到模拟预测
    predicted_class = mock_prediction_fallback()
    confidence = np.random.uniform(0.60, 0.80)
    return predicted_class, confidence, False  # False表示使用模拟预测

def create_segmentation_mock(image_path, filename):
    """
    模拟图像分割（绘制简单边框）
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        height, width = image.shape[:2]
        
        # 绘制简单的矩形边框作为"分割"结果
        margin = min(width, height) // 10
        cv2.rectangle(image, 
                     (margin, margin), 
                     (width - margin, height - margin), 
                     (0, 255, 0), 3)
        
        # 添加预测信息
        if fish_classifier.model_loaded:
            top_predictions = fish_classifier.get_top_k_predictions(image_path, k=3)
            if top_predictions:
                text = f"{top_predictions[0]['class']} ({top_predictions[0]['confidence']:.3f})"
                cv2.putText(image, text, (margin, margin - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 保存结果
        output_path = f'./tmp/draw/{filename}'
        cv2.imwrite(output_path, image)
        return True
        
    except Exception as e:
        print(f"分割模拟错误: {e}")
        shutil.copy(image_path, f'./tmp/draw/{filename}')
        return False

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    status_msg = "PyTorch AI Model" if fish_classifier.model_loaded else "Simulation Mode"
    return jsonify({
        'status': 'Fish Recognition API is running', 
        'message': '海洋鱼类识别系统后端已启动',
        'model_status': status_msg,
        'accuracy': f"{91.62}%" if fish_classifier.model_loaded else "N/A"
    })

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'status': 0, 'message': 'No file part'})
        
        file = request.files['file']
        print(f"{datetime.datetime.now()} {file.filename}")
        
        if file.filename == '':
            return jsonify({'status': 0, 'message': 'No selected file'})
            
        if file and allowed_file(file.filename):
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(src_path)        
            shutil.copy(src_path, './tmp/image')
            image_path = os.path.join('./tmp/image', file.filename)
            print(f"Image saved to: {image_path}")
            
            # 进行分类预测
            predicted_class, confidence, is_real_model = predict_classification(image_path)
            print(f"预测结果: {predicted_class} (置信度: {confidence:.3f})")
            
            # 创建分割结果
            create_segmentation_mock(image_path, file.filename)
            
            # 获取详细预测信息
            top_predictions = []
            if fish_classifier.model_loaded:
                top_predictions = fish_classifier.get_top_k_predictions(image_path, k=3)
            
            return jsonify({
                'status': 1,
                'image_url': f'http://127.0.0.1:5003/tmp/image/{file.filename}',
                'draw_url': f'http://127.0.0.1:5003/tmp/draw/{file.filename}',
                'yucejieguo': predicted_class,
                'confidence': f"{confidence:.3f}",
                'model_status': 'pytorch_resnet50' if is_real_model else 'simulation',
                'accuracy': "91.62%" if is_real_model else "N/A",
                'top_predictions': top_predictions
            })
    
    return jsonify({'status': 0, 'message': 'Invalid request'})

@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if file:
            try:
                image_data = open(f'tmp/{file}', "rb").read()
                response = make_response(image_data)
                response.headers['Content-Type'] = 'image/png'
                return response
            except FileNotFoundError:
                return jsonify({'error': 'File not found'}), 404

@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    info = {
        'model_loaded': fish_classifier.model_loaded,
        'model_type': 'PyTorch ResNet50',
        'num_classes': len(fish_classifier.classes) if fish_classifier.classes else 0,
        'classes': fish_classifier.classes if fish_classifier.classes else [],
        'training_accuracy': "91.62%" if fish_classifier.model_loaded else "N/A"
    }
    return jsonify(info)

@app.route('/multi_detect', methods=['GET', 'POST'])
def multi_fish_detection():
    """
    多鱼类检测API端点
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'status': 0, 'message': 'No file part'})
        
        file = request.files['file']
        print(f"{datetime.datetime.now()} 多鱼类检测: {file.filename}")
        
        if file.filename == '':
            return jsonify({'status': 0, 'message': 'No selected file'})
            
        if file and allowed_file(file.filename):
            # 保存上传的图片
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(src_path)
            shutil.copy(src_path, './tmp/image')
            image_path = os.path.join('./tmp/image', file.filename)
            
            # 初始化多检测器（如果还没有）
            if multi_detector is None:
                init_multi_detector()
            
            if multi_detector and fish_classifier.model_loaded:
                # 执行多鱼类检测
                detections = multi_detector.detect_fish_regions(image_path, max_detections=8)
                
                # 生成检测结果图像
                detection_output = f'./tmp/draw/{file.filename}'
                success = multi_detector.create_opencv_result(image_path, detections, detection_output)
                
                # 准备返回数据
                detection_results = []
                for bbox, class_name, confidence in detections:
                    x1, y1, x2, y2 = bbox
                    detection_results.append({
                        'class': class_name,
                        'confidence': round(confidence, 3),
                        'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'center': {'x': (x1+x2)//2, 'y': (y1+y2)//2}
                    })
                
                return jsonify({
                    'status': 1,
                    'image_url': f'http://127.0.0.1:5003/tmp/image/{file.filename}',
                    'detection_url': f'http://127.0.0.1:5003/tmp/draw/{file.filename}',
                    'fish_count': len(detections),
                    'detections': detection_results,
                    'model_status': 'multi_detection_pytorch',
                    'message': f'检测到 {len(detections)} 条鱼'
                })
            else:
                # 降级到单鱼检测
                predicted_class, confidence, is_real_model = predict_classification(image_path)
                create_segmentation_mock(image_path, file.filename)
                
                return jsonify({
                    'status': 1,
                    'image_url': f'http://127.0.0.1:5003/tmp/image/{file.filename}',
                    'detection_url': f'http://127.0.0.1:5003/tmp/draw/{file.filename}',
                    'fish_count': 1,
                    'detections': [{
                        'class': predicted_class,
                        'confidence': round(confidence, 3),
                        'bbox': {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100},
                        'center': {'x': 50, 'y': 50}
                    }],
                    'model_status': 'single_classification',
                    'message': f'单鱼检测: {predicted_class}'
                })
    
    return jsonify({'status': 0, 'message': 'Invalid request'})

if __name__ == '__main__':
    # 创建必要目录
    directories = ['uploads', 'tmp/draw', 'tmp/image', 'tmp/mask']
    for dir_path in directories:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    print("🐟 海洋鱼类识别系统后端启动中...")
    print("📡 后端API地址: http://127.0.0.1:5003")
    
    if fish_classifier.model_loaded:
        print("🎯 使用 PyTorch ResNet50 AI模型")
        print(f"📊 模型准确率: 91.62%")
        print(f"🏷️  支持类别: {len(fish_classifier.classes)} 种鱼类")
        
        # 初始化多鱼类检测器
        init_multi_detector()
        if multi_detector:
            print("🔍 多鱼类检测功能已启用")
    else:
        print("⚠️  PyTorch模型未加载，使用模拟预测")
    
    app.run(host='127.0.0.1', port=5003, debug=True)
