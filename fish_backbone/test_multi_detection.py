"""
测试多鱼类检测功能
"""

import sys
import os
sys.path.append('.')

from app_pytorch import fish_classifier
from multi_fish_detector import MultiFishDetector
import cv2

def test_multi_detection():
    """测试多鱼类检测"""
    
    print("🧪 测试多鱼类检测功能")
    
    if not fish_classifier.model_loaded:
        print("❌ 分类模型未加载")
        return
    
    # 创建检测器，使用较低的阈值
    detector = MultiFishDetector(
        fish_classifier, 
        confidence_threshold=0.3,  # 降低阈值
        overlap_threshold=0.4
    )
    
    # 测试图片
    test_images = [
        "./uploads/fish_000004210001_00019.png",
        "./uploads/fish_000005829592_03456.png",
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  图片不存在: {image_path}")
            continue
            
        print(f"\n🔍 检测图片: {image_path}")
        
        # 执行检测
        detections = detector.detect_fish_regions(image_path, max_detections=6)
        
        if detections:
            print(f"✅ 检测到 {len(detections)} 条鱼:")
            for i, (bbox, class_name, confidence) in enumerate(detections):
                x1, y1, x2, y2 = bbox
                print(f"  鱼 {i+1}: {class_name} (置信度: {confidence:.3f}) 位置: ({x1},{y1},{x2},{y2})")
            
            # 生成可视化结果
            output_name = os.path.basename(image_path).replace('.png', '_multi_detect.jpg')
            output_path = f"./tmp/{output_name}"
            
            success = detector.create_opencv_result(image_path, detections, output_path)
            if success:
                print(f"📊 检测结果已保存: {output_path}")
        else:
            print("❌ 未检测到鱼类")
            
            # 尝试单鱼检测作为对比
            try:
                pred_class, confidence = fish_classifier.predict(image_path)
                print(f"🎯 单鱼检测结果: {pred_class} (置信度: {confidence:.3f})")
            except Exception as e:
                print(f"❌ 单鱼检测也失败: {e}")

if __name__ == "__main__":
    test_multi_detection()
