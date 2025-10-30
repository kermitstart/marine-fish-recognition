#!/bin/bash

# 海洋鱼类识别项目 - GitHub推送脚本
# 使用说明：
# 1. 在GitHub上创建一个新仓库 (如: marine-fish-recognition)
# 2. 将下面的 YOUR_USERNAME 和 YOUR_REPOSITORY 替换为你的GitHub用户名和仓库名
# 3. 运行 chmod +x push_to_github.sh
# 4. 运行 ./push_to_github.sh

echo "=== 海洋鱼类识别项目 - GitHub推送脚本 ==="
echo ""

# 请在这里修改为你的GitHub仓库信息
read -p "请输入你的GitHub用户名: " GITHUB_USERNAME
read -p "请输入仓库名 (默认: marine-fish-recognition): " REPOSITORY_NAME

# 设置默认仓库名
if [ -z "$REPOSITORY_NAME" ]; then
    REPOSITORY_NAME="marine-fish-recognition"
fi

echo "GitHub用户名: $GITHUB_USERNAME"
echo "仓库名: $REPOSITORY_NAME"
echo ""

# 检查是否输入了用户名
if [ -z "$GITHUB_USERNAME" ]; then
    echo "❌ 错误: GitHub用户名不能为空"
    exit 1
fi

echo "1. 检查git配置..."
git config --global user.name || echo "请设置: git config --global user.name 'Your Name'"
git config --global user.email || echo "请设置: git config --global user.email 'your.email@example.com'"

echo ""
echo "2. 添加远程仓库..."
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/$GITHUB_USERNAME/$REPOSITORY_NAME.git

echo ""
echo "3. 检查当前文件状态..."
git status

echo ""
echo "4. 添加所有文件到git..."
git add .

echo ""
echo "5. 创建提交..."
git commit -m "Initial commit: Marine Fish Recognition Project

Features:
- 支持22种海洋鱼类识别
- 提供完整、紧凑、mini三种数据集
- Google Colab一键训练notebook
- PyTorch深度学习模型
- Web界面预测功能
- 数据增强和模型优化
- 自动化训练和部署脚本

数据集:
- 完整数据集: 22个鱼类，每类100+样本
- 紧凑数据集: 22个鱼类，每类50样本 
- Mini数据集: 22个鱼类，每类10样本

Colab支持:
- 自动从GitHub克隆项目
- 自动安装依赖
- 智能数据集选择
- 完整训练流程
- 结果可视化
- 模型部署指南
- 自动保存到Google Drive"

echo ""
echo "6. 推送到GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "✅ 项目已成功推送到GitHub!"
echo ""
echo "🔗 你的项目地址: https://github.com/$GITHUB_USERNAME/$REPOSITORY_NAME"
echo ""
echo "📝 下一步操作:"
echo "1. 打开你的GitHub仓库地址，确认文件已上传成功"
echo "2. 复制仓库地址，用于Colab notebook中"
echo "3. 在Google Colab中打开 colab_fish_training.ipynb"
echo "4. 修改第一个代码cell中的GitHub仓库地址"
echo "5. 开始愉快的训练吧！"
echo ""
echo "💡 Colab使用说明请参考: COLAB_SETUP.md"
