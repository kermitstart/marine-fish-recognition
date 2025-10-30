"""
多鱼类目标检测系统
使用滑窗检测 + 分类模型实现多目标识别
"""

import torch
import cv2
import numpy as np
from PIL import Image
import json
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class MultiFishDetector:
    """
    多鱼类目标检测器
    结合滑窗检测和分类模型
    """
    
    def __init__(self, classifier, confidence_threshold=0.6, overlap_threshold=0.3):
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold
        
        # 检测窗口大小（多尺度）
        # 添加更多尺度的检测窗口
        self.window_sizes = [
            (48, 48),  # 更小的鱼
            (64, 64),
            (96, 96),
            (128, 128),
            (160, 160),
            (192, 192),  # 更大的鱼
        ]

        # self.window_sizes = [
        #     (64, 64),     # 超小鱼
        #     (96, 96),     # 小鱼
        #     (128, 128),   # 中等鱼
        #     (160, 160),   # 大鱼
        # ]
        
        # 滑窗步长
        self.stride = 32
        
    def detect_fish_regions(self, image_path, max_detections=10):
        """
        检测图片中的多条鱼
        返回: [(bbox, class_name, confidence), ...]
        """
        try:
            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            height, width = image.shape[:2]
            detections = []
            
            print(f"🔍 开始多鱼类检测... 图像尺寸: {width}x{height}")
            
            # 多尺度滑窗检测
            for window_w, window_h in self.window_sizes:
                # 跳过过大的检测窗口
                if window_w > width or window_h > height:
                    continue
                    
                print(f"  🔎 检测窗口: {window_w}x{window_h}")
                
                # 滑窗遍历
                for y in range(0, height - window_h + 1, self.stride):
                    for x in range(0, width - window_w + 1, self.stride):
                        # 提取窗口区域
                        window = image[y:y+window_h, x:x+window_w]
                        
                        # 检查窗口是否有足够的内容
                        if self._is_valid_window(window):
                            # 保存临时图片进行分类
                            temp_path = './tmp/temp_window.jpg'
                            cv2.imwrite(temp_path, window)
                            
                            # 使用分类器预测
                            predicted_class, confidence = self.classifier.predict(temp_path)
                            
                            # 如果置信度足够高，记录检测结果
                            if confidence > self.confidence_threshold:
                                bbox = (x, y, x + window_w, y + window_h)
                                detections.append((bbox, predicted_class, confidence))
            
            # 非极大值抑制，去除重叠检测
            filtered_detections = self._non_max_suppression(detections)
            
            # 限制检测数量
            filtered_detections = filtered_detections[:max_detections]
            
            print(f"✅ 检测完成，发现 {len(filtered_detections)} 条鱼")
            return filtered_detections
            
        except Exception as e:
            print(f"❌ 多鱼类检测失败: {e}")
            return []
    
    def _is_valid_window(self, window):
        """
        检查窗口是否包含有效内容
        """
        if window.size == 0:
            return False
        
        # 检查方差（避免空白区域）
        gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        
        # 如果方差太小，说明区域过于单调
        return variance > 100
    
    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的交并比(IoU)
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # 计算各自面积
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # 计算并集
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _non_max_suppression(self, detections):
        """
        非极大值抑制，去除重叠检测
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x[2], reverse=True)
        
        filtered = []
        
        for current in detections:
            current_bbox, current_class, current_conf = current
            
            # 检查是否与已选择的检测重叠
            is_overlap = False
            for selected in filtered:
                selected_bbox, selected_class, selected_conf = selected
                
                iou = self._calculate_iou(current_bbox, selected_bbox)
                if iou > self.overlap_threshold:
                    is_overlap = True
                    break
            
            if not is_overlap:
                filtered.append(current)
        
        return filtered
    
    def visualize_detections(self, image_path, detections, output_path):
        """
        可视化检测结果
        """
        try:
            # 读取原图
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # 转换为RGB用于matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(image_rgb)
            
            # 颜色列表
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
            
            for i, (bbox, class_name, confidence) in enumerate(detections):
                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]
                
                # 绘制边界框
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # 添加标签
                label = f"{class_name}\n{confidence:.3f}"
                ax.text(x1, y1-10, label, fontsize=10, color=color, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax.set_title(f'多鱼类检测结果 - 发现 {len(detections)} 条鱼', fontsize=14)
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return True
            
        except Exception as e:
            print(f"可视化失败: {e}")
            return False

    def create_opencv_result(self, image_path, detections, output_path):
        """
        使用OpenCV创建检测结果图像
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # 颜色列表 (BGR格式)
            colors = [
                (0, 0, 255),    # 红色
                (255, 0, 0),    # 蓝色
                (0, 255, 0),    # 绿色
                (0, 255, 255),  # 黄色
                (255, 0, 255),  # 紫色
                (0, 165, 255),  # 橙色
                (255, 192, 203), # 粉色
                (255, 255, 0),  # 青色
            ]
            
            for i, (bbox, class_name, confidence) in enumerate(detections):
                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                
                # 添加标签背景
                label = f"{class_name}: {confidence:.3f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # 绘制标签背景
                cv2.rectangle(image, (x1, y1-text_h-10), (x1+text_w, y1), color, -1)
                
                # 绘制文字
                cv2.putText(image, label, (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 添加总结信息
            summary = f"Detected {len(detections)} fish"
            cv2.putText(image, summary, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imwrite(output_path, image)
            return True
            
        except Exception as e:
            print(f"OpenCV可视化失败: {e}")
            return False


def test_multi_fish_detection():
    """
    测试多鱼类检测功能
    """
    print("🧪 测试多鱼类检测功能")
    
    # 这里需要加载已有的分类器
    # 假设fish_classifier已经初始化
    from app_pytorch import fish_classifier
    
    if not fish_classifier.model_loaded:
        print("❌ 分类模型未加载")
        return
    
    detector = MultiFishDetector(fish_classifier, confidence_threshold=0.5)
    
    # 测试图片路径
    test_image = "./uploads/fish_000004210001_00019.png"
    
    # 执行检测
    detections = detector.detect_fish_regions(test_image)
    
    if detections:
        print(f"✅ 检测结果:")
        for i, (bbox, class_name, confidence) in enumerate(detections):
            print(f"  鱼 {i+1}: {class_name} (置信度: {confidence:.3f}) 位置: {bbox}")
        
        # 生成可视化结果
        output_path = "./tmp/multi_fish_detection.jpg"
        detector.create_opencv_result(test_image, detections, output_path)
        print(f"📊 检测结果已保存到: {output_path}")
    else:
        print("❌ 未检测到鱼类")


if __name__ == "__main__":
    test_multi_fish_detection()
