# -*- coding: utf-8 -*-
import datetime
import logging as rel_log
import os
import shutil
from datetime import timedelta
from flask import *
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

UPLOAD_FOLDER = r'./uploads'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)

# 解决缓存刷新问题
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)

# 添加header解决跨域
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def hello_world():
    return jsonify({'status': 'Fish Recognition API is running', 'message': '海洋鱼类识别系统后端已启动'})

# 鱼类名称翻译字典
def get_chinese_fish_name(english_name):
    fish_translations = {
        "Abudefduf_vaigiensis": "双带豆娘鱼",
        "Acanthurus_nigrofuscus": "黑斑刺尾鱼", 
        "Amphiprion_clarkia": "克拉氏小丑鱼",
        "Balistapus_undulates": "疣鳞鲀",
        "Canthigaster_valentine": "情人刺鲀",
        "Chaetodon_lunulatus": "月斑蝴蝶鱼",
        "Chaetodon_trifascialis": "三带蝴蝶鱼",
        "Chromis_chrysura": "黄尾光鳃鱼",
        "Dascyllus_reticulatus": "网纹三斑鱼",
        "Hemigymnus_fasciatus": "条纹半裸鱼",
        "Hemigymnus_melapterus": "黑鳍半裸鱼",
        "Lutjanus_fulvus": "黄鲷",
        "Myripristis_kuntee": "昆氏多棘鱼",
        "Ncoglyphidodon_nigroris": "黑边刻齿鱼",
        "Neoniphon_samara": "萨马拉新鳞鱼",
        "Pempheris_vanicolensis": "瓦尼科伦斯玻璃鱼",
        "Plectroglyphidodon_dickii": "迪氏刻齿鱼",
        "Pomacentrus_moluccensis": "摩鹿加雀鲷",
        "Scaridae": "鹦鹉鱼科",
        "Scolopsis_bilineata": "双线鱼",
        "Siganus_fuscescens": "褐蓝子鱼",
        "Zanclus_cornutus": "角蝶鱼",
        "Zebrasoma_scopas": "褐刺尾鱼"
    }
    return fish_translations.get(english_name, english_name)

# 模拟预测函数（当没有真实模型时）
def mock_prediction(image_path):
    # 模拟鱼类识别结果（返回英文名称）
    fish_names = [
        "Abudefduf_vaigiensis", "Acanthurus_nigrofuscus", "Amphiprion_clarkia",
        "Balistapus_undulates", "Canthigaster_valentine", "Chaetodon_lunulatus"
    ]
    import random
    english_name = random.choice(fish_names)
    # 返回英文名称用于显示
    return english_name

# 模拟多鱼检测函数
def mock_multi_detection(image_path):
    import random
    # 英文鱼类名称列表
    english_fish_names = [
        "Abudefduf_vaigiensis", "Acanthurus_nigrofuscus", "Amphiprion_clarkia",
        "Balistapus_undulates", "Canthigaster_valentine", "Chaetodon_lunulatus",
        "Chaetodon_trifascialis", "Chromis_chrysura", "Dascyllus_reticulatus"
    ]
    
    # 随机生成1-4条鱼的检测结果
    fish_count = random.randint(1, 4)
    detections = []
    
    for i in range(fish_count):
        english_name = random.choice(english_fish_names)
        
        detections.append({
            "class": english_name,  # 使用英文名称
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "bbox": [
                random.randint(10, 100),  # x
                random.randint(10, 100),  # y
                random.randint(150, 300), # width
                random.randint(100, 200)  # height
            ]
        })
    
    return fish_count, detections

# 使用PIL绘制支持中文的检测框
def draw_single_fish_detection(image_path, fish_name, output_path):
    try:
        # 使用PIL读取图像以支持中文
        from PIL import Image, ImageDraw, ImageFont
        import cv2
        import numpy as np
        
        # 读取图像
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            return False
            
        # 转换为PIL图像
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        
        h, w = pil_image.size[1], pil_image.size[0]
        
        # 创建一个居中的检测框（模拟检测结果）
        margin = min(w, h) // 6
        x1, y1 = margin, margin
        x2, y2 = w - margin, h - margin
        
        # 绘制检测框
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        
        # 尝试加载中文字体，如果失败则使用默认字体
        try:
            # 使用系统中文字体
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        except:
            try:
                # 尝试PingFang中文字体
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 28)
            except:
                try:
                    # 尝试Arial Unicode支持中文
                    font = ImageFont.truetype("/System/Library/Fonts/Arial Unicode.ttf", 28)
                except:
                    # 使用默认字体，但会有中文显示问题
                    font = ImageFont.load_default()
        
        # 绘制标签背景和文本
        label = fish_name
        print(f"绘制单鱼检测框标签: {label}")  # 调试信息
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制标签背景
        draw.rectangle([x1, y1 - text_height - 10, x1 + text_width + 10, y1], fill=(0, 255, 0))
        
        # 绘制标签文本
        draw.text((x1 + 5, y1 - text_height - 5), label, fill=(0, 0, 0), font=font)
        
        # 转换回OpenCV格式并保存
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, opencv_image)
        return True
        
    except Exception as e:
        print(f"绘制检测框错误: {e}")
        return False

# 使用PIL绘制支持中文的多鱼检测框
def draw_multi_fish_detection(image_path, detections, output_path):
    try:
        from PIL import Image, ImageDraw, ImageFont
        import cv2
        import numpy as np
        
        # 读取图像
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            return False
            
        # 转换为PIL图像
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        
        h, w = pil_image.size[1], pil_image.size[0]
        
        # 尝试加载中文字体
        try:
            # 使用系统中文字体
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            try:
                # 尝试PingFang中文字体
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 24)
            except:
                try:
                    # 尝试Arial Unicode支持中文
                    font = ImageFont.truetype("/System/Library/Fonts/Arial Unicode.ttf", 24)
                except:
                    # 使用默认字体，但会有中文显示问题
                    font = ImageFont.load_default()
        
        # 为每个检测结果绘制框
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, detection in enumerate(detections):
            color = colors[i % len(colors)]
            
            # 获取边界框（如果没有真实坐标，生成模拟坐标）
            if 'bbox' in detection:
                x, y, bbox_w, bbox_h = detection['bbox']
                x1, y1, x2, y2 = x, y, x + bbox_w, y + bbox_h
            else:
                # 生成随机位置的框
                box_w, box_h = w // 4, h // 4
                x1 = (i * w // len(detections)) % (w - box_w)
                y1 = (i * h // 3) % (h - box_h)
                x2, y2 = x1 + box_w, y1 + box_h
            
            # 绘制检测框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 添加标签
            label = f"{detection['class']} {detection['confidence']:.0%}"
            print(f"绘制检测框标签: {label}")  # 调试信息
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # 绘制标签背景
            draw.rectangle([x1, y1 - text_height - 10, x1 + text_width + 10, y1], fill=color)
            
            # 绘制标签文本
            draw.text((x1 + 5, y1 - text_height - 5), label, fill=(255, 255, 255), font=font)
        
        # 转换回OpenCV格式并保存
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, opencv_image)
        return True
        
    except Exception as e:
        print(f"绘制多鱼检测框错误: {e}")
        return False

#update file
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 支持重新检测已存在的图片
        redetect_file = request.args.get('redetect')
        if redetect_file:
            image_path = os.path.join('./tmp/image', redetect_file)
            if os.path.exists(image_path):
                print(f"重新检测图片: {image_path}")
                yucejieguo = mock_prediction(image_path)
                print(f"单鱼重新检测结果: {yucejieguo}")
                
                # 绘制单鱼检测框
                draw_output_path = f'./tmp/draw/{redetect_file}'
                success = draw_single_fish_detection(image_path, yucejieguo, draw_output_path)
                
                if not success:
                    shutil.copy(image_path, draw_output_path)
                
                return jsonify({
                    'status': 1,
                    'image_url': f'http://127.0.0.1:5003/tmp/image/{redetect_file}',
                    'draw_url': f'http://127.0.0.1:5003/tmp/draw/{redetect_file}',
                    'yucejieguo': yucejieguo,
                    'fish_name': yucejieguo,
                    'confidence': 0.85
                })
        
        if 'file' not in request.files:
            return jsonify({'status': 0, 'message': 'No file part'})
        
        file = request.files['file']
        print(datetime.datetime.now(), file.filename)
        
        if file.filename == '':
            return jsonify({'status': 0, 'message': 'No selected file'})
            
        if file and allowed_file(file.filename):
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(src_path)        
            shutil.copy(src_path, './tmp/image')
            image_path = os.path.join('./tmp/image', file.filename)
            print(f"Image saved to: {image_path}")
            
            # 使用模拟预测（因为还没有真实模型）
            yucejieguo = mock_prediction(image_path)
            print(f"预测结果: {yucejieguo}")
            
            # 绘制单鱼检测框
            draw_output_path = f'./tmp/draw/{file.filename}'
            success = draw_single_fish_detection(image_path, yucejieguo, draw_output_path)
            
            if not success:
                # 如果绘制失败，复制原图
                shutil.copy(image_path, draw_output_path)
            
            return jsonify({
                'status': 1,
                'image_url': f'http://127.0.0.1:5003/tmp/image/{file.filename}',
                'draw_url': f'http://127.0.0.1:5003/tmp/draw/{file.filename}',
                'yucejieguo': yucejieguo,
                'fish_name': yucejieguo,
                'confidence': 0.85
            })
    
    return jsonify({'status': 0, 'message': 'Invalid request'})

# 多鱼检测API
@app.route('/multi_detect', methods=['GET', 'POST'])
def multi_detect_file():
    if request.method == 'POST':
        # 支持重新检测已存在的图片
        redetect_file = request.args.get('redetect')
        if redetect_file:
            image_path = os.path.join('./tmp/image', redetect_file)
            if os.path.exists(image_path):
                print(f"重新检测图片: {image_path}")
                fish_count, detections = mock_multi_detection(image_path)
                print(f"多鱼重新检测结果: {fish_count} 条鱼, {detections}")
                
                # 绘制多鱼检测框
                draw_output_path = f'./tmp/draw/{redetect_file}'
                success = draw_multi_fish_detection(image_path, detections, draw_output_path)
                
                if not success:
                    shutil.copy(image_path, draw_output_path)
                
                return jsonify({
                    'status': 1,
                    'image_url': f'http://127.0.0.1:5003/tmp/image/{redetect_file}',
                    'detection_url': f'http://127.0.0.1:5003/tmp/draw/{redetect_file}',
                    'fish_count': fish_count,
                    'detections': detections,
                    'message': f'重新检测到 {fish_count} 条鱼类'
                })
        
        if 'file' not in request.files:
            return jsonify({'status': 0, 'message': 'No file part'})
        
        file = request.files['file']
        print(datetime.datetime.now(), "Multi-detect:", file.filename)
        
        if file.filename == '':
            return jsonify({'status': 0, 'message': 'No selected file'})
            
        if file and allowed_file(file.filename):
            src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(src_path)        
            shutil.copy(src_path, './tmp/image')
            image_path = os.path.join('./tmp/image', file.filename)
            print(f"Multi-detect image saved to: {image_path}")
            
            # 使用模拟多鱼检测
            fish_count, detections = mock_multi_detection(image_path)
            print(f"多鱼检测结果: {fish_count} 条鱼, {detections}")
            
            # 绘制多鱼检测框
            draw_output_path = f'./tmp/draw/{file.filename}'
            success = draw_multi_fish_detection(image_path, detections, draw_output_path)
            
            if not success:
                # 如果绘制失败，复制原图
                shutil.copy(image_path, draw_output_path)
            
            return jsonify({
                'status': 1,
                'image_url': f'http://127.0.0.1:5003/tmp/image/{file.filename}',
                'detection_url': f'http://127.0.0.1:5003/tmp/draw/{file.filename}',
                'fish_count': fish_count,
                'detections': detections,
                'message': f'检测到 {fish_count} 条鱼类'
            })
    
    return jsonify({'status': 0, 'message': 'Invalid request'})

# show photo
@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            try:
                image_data = open(f'tmp/{file}',"rb").read()
                response = make_response(image_data)
                response.headers['Content-Type'] = 'image/png'
                return response
            except FileNotFoundError:
                return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    files = ['uploads', 'tmp/draw','tmp/image', 'tmp/mask', 'tmp/uploads']
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    
    print("🐟 海洋鱼类识别系统后端启动中...")
    print("📡 后端API地址: http://127.0.0.1:5003")
    print("📝 注意: 当前使用模拟预测，需要下载真实模型以获得准确结果")
    
    app.run(host='127.0.0.1', port=5003, debug=True)
