#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试中文字体绘制
"""
from PIL import Image, ImageDraw, ImageFont
import os

def test_chinese_fonts():
    # 创建一个测试图像
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # 测试文本
    test_text = "双带豆娘鱼 85%"
    
    # 测试不同字体
    fonts_to_test = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/PingFang.ttc", 
        "/System/Library/Fonts/Arial Unicode.ttf",
    ]
    
    y_pos = 20
    
    for font_path in fonts_to_test:
        try:
            font = ImageFont.truetype(font_path, 24)
            font_name = os.path.basename(font_path)
            draw.text((20, y_pos), f"{font_name}: {test_text}", fill='black', font=font)
            print(f"✅ 成功加载字体: {font_name}")
            y_pos += 40
        except Exception as e:
            print(f"❌ 无法加载字体 {font_path}: {e}")
    
    # 测试默认字体
    try:
        default_font = ImageFont.load_default()
        draw.text((20, y_pos), f"Default: {test_text}", fill='red', font=default_font)
        print("✅ 成功加载默认字体")
        y_pos += 40
    except Exception as e:
        print(f"❌ 无法加载默认字体: {e}")
    
    # 保存测试图像
    test_path = './font_test.png'
    img.save(test_path)
    print(f"📷 测试图像已保存到: {test_path}")
    
    return test_path

if __name__ == "__main__":
    test_chinese_fonts()
