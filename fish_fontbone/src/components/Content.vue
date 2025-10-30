<!-- 模板 -->
<template>
  <div id="Content">
    
    <el-dialog
      title="🐟 AI识别处理中"
      :visible.sync="dialogTableVisible"
      :show-close="false"
      :close-on-press-escape="false"
      :append-to-body="true"
      :close-on-click-modal="false"
      :center="true"
      width="400px"
    >
      <div style="text-align: center; padding: 20px;">
        <el-progress 
          :percentage="percentage" 
          :stroke-width="8"
          color="#409EFF"
        ></el-progress>
        <p style="margin-top: 15px; color: #666;">AI正在分析图像，请耐心等待...</p>
      </div>
    </el-dialog>

    <!-- 主容器 -->
    <div class="main-container">
      <!-- 侧边栏 - 检测结果显示 -->
      <div class="sidebar">
        <!-- AI模型信息卡片 -->
        <el-card class="info-card model-info" shadow="hover">
          <div slot="header" class="card-header">
            <i class="el-icon-cpu"></i>
            <span>AI模型信息</span>
          </div>
          <div class="model-stats">
            <div class="stat-item">
              <span class="label">模型类型</span>
              <span class="value">PyTorch ResNet50</span>
            </div>
            <div class="stat-item">
              <span class="label">准确率</span>
              <span class="value highlight">91.62%</span>
            </div>
            <div class="stat-item">
              <span class="label">支持类别</span>
              <span class="value">23种海洋鱼类</span>
            </div>
          </div>
        </el-card>

        <!-- 检测结果卡片 -->
        <el-card class="info-card result-info" shadow="hover">
          <div slot="header" class="card-header">
            <i class="el-icon-data-analysis"></i>
            <span>检测结果</span>
          </div>
          <div v-if="fishCount > 0" class="detection-results">
            <div class="result-summary">
              <span class="fish-count">{{ fishCount }}</span>
              <span class="count-label">{{ detectionMode === 'multi' ? '条鱼类' : '种鱼类' }}</span>
            </div>
            
            <!-- 单鱼检测结果 -->
            <div v-if="detectionMode === 'single' && detections.length > 0" class="single-fish-result">
              <div class="fish-card">
                <div class="fish-icon">🐟</div>
                <div class="fish-details">
                  <div class="fish-name">{{ detections[0].class }}</div>
                  <div class="confidence">置信度: {{ (detections[0].confidence * 100).toFixed(1) }}%</div>
                </div>
              </div>
            </div>
            
            <!-- 多鱼检测结果 -->
            <div v-else-if="detectionMode === 'multi'" class="detection-list">
              <div 
                v-for="(detection, index) in detections.slice(0, 3)" 
                :key="index"
                class="detection-item"
              >
                <div class="detection-index">{{ index + 1 }}</div>
                <div class="detection-info">
                  <div class="fish-name">{{ detection.class }}</div>
                  <div class="confidence">置信度: {{ (detection.confidence * 100).toFixed(1) }}%</div>
                </div>
              </div>
              <div v-if="detections.length > 3" class="more-results">
                还有 {{ detections.length - 3 }} 个检测结果...
              </div>
            </div>
          </div>
          <div v-else class="no-result">
            <i class="el-icon-picture-outline"></i>
            <p>暂无检测结果</p>
            <p class="tip">上传图片开始AI识别</p>
          </div>
        </el-card>
      </div>

      <!-- 主内容区域 -->
      <div class="main-content">
        <!-- 控制面板 -->
        <div class="control-panel">
          <el-card shadow="never" class="control-card">
            <div class="mode-selector">
              <h3 class="section-title">
                <i class="el-icon-s-tools"></i>
                检测模式
              </h3>
              <el-radio-group v-model="detectionMode" size="medium" class="mode-buttons">
                <el-radio-button label="single" class="mode-btn">
                  <i class="el-icon-view"></i>
                  单鱼检测
                </el-radio-button>
                <el-radio-button label="multi" class="mode-btn">
                  <i class="el-icon-s-grid"></i>
                  多鱼检测
                </el-radio-button>
              </el-radio-group>
            </div>
          </el-card>
        </div>

        <!-- 图片展示区域 -->
        <div class="image-display">
          <!-- 原图区域 -->
          <div class="image-section">
            <el-card class="image-card original-image" shadow="hover">
              <div slot="header" class="image-header">
                <i class="el-icon-picture"></i>
                <span>原始图片</span>
              </div>
              <div class="image-content">
                <div
                  v-loading="loading"
                  element-loading-text="正在上传..."
                  element-loading-spinner="el-icon-loading"
                  element-loading-background="rgba(0, 0, 0, 0.1)"
                  class="image-wrapper"
                >
                  <el-image
                    v-if="url_1"
                    :src="url_1"
                    class="display-image"
                    :preview-src-list="srcList1"
                    fit="contain"
                  ></el-image>
                  <div v-else class="upload-area" @click="true_upload">
                    <div class="upload-content">
                      <i class="el-icon-upload upload-icon"></i>
                      <p class="upload-text">点击或拖拽上传图片</p>
                      <p class="upload-hint">支持 PNG、JPG 格式</p>
                    </div>
                    <input
                      ref="upload"
                      style="display: none"
                      name="file"
                      type="file"
                      accept="image/*"
                      @change="update"
                    />
                  </div>
                </div>
              </div>
            </el-card>
          </div>

          <!-- 检测结果图像区域 -->
          <div class="image-section">
            <el-card class="image-card result-image" shadow="hover">
              <div slot="header" class="image-header">
                <i class="el-icon-data-analysis"></i>
                <span>{{ detectionMode === 'multi' ? '多鱼检测结果' : '识别结果' }}</span>
              </div>
              <div class="image-content">
                <div
                  v-loading="loading"
                  element-loading-text="AI识别中..."
                  element-loading-spinner="el-icon-loading"
                  element-loading-background="rgba(0, 0, 0, 0.1)"
                  class="image-wrapper"
                >
                  <el-image
                    v-if="url_2"
                    :src="url_2"
                    class="display-image"
                    :preview-src-list="srcList2"
                    fit="contain"
                  ></el-image>
                  <div v-else class="waiting-area">
                    <div class="waiting-content">
                      <i class="el-icon-cpu waiting-icon"></i>
                      <p class="waiting-text">{{ url_1 ? '点击下方按钮开始检测' : '等待AI识别结果' }}</p>
                    </div>
                  </div>
                </div>
              </div>
            </el-card>
          </div>
        </div>

        <!-- 操作按钮区域 -->
        <div class="action-panel" v-if="url_1">
          <el-button
            type="success"
            icon="el-icon-search"
            size="large"
            class="action-button detect-button"
            @click="reDetect"
            :disabled="loading"
          >
            {{ detectionMode === 'multi' ? '开始多鱼检测' : '开始单鱼检测' }}
          </el-button>
          <el-button
            type="primary"
            icon="el-icon-refresh"
            size="large"
            class="action-button"
            @click="true_upload"
          >
            重新上传图片
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>




<script>

import axios from "axios";

export default {
  name: "Content",
  data() {
    return {
      server_url: "http://127.0.0.1:5003",
      detectionMode: "single", // 检测模式：single(单鱼) 或 multi(多鱼)
      active: 0,
      centerDialogVisible: true,
      url_1: "",
      url_2: "",
      textarea: "",
      srcList1: [],
      srcList2: [],
      url: "",
      visible: false,
      wait_return: "等待返回",
      wait_upload: "等待上传",
      yucejieguo: "",
      loading: false,
      table: false,
      isNav: false,
      showbutton: true,
      percentage: 0,
      fullscreenLoading: false,
      opacitys: {
        opacity: 0,
      },
      dialogTableVisible: false,
      fishCount: 0, // 检测到的鱼类数量
      detections: [], // 检测结果列表
    };
  },
  watch: {
    // 监听检测模式变化，切换时仅清空结果
    detectionMode(newMode, oldMode) {
      if (newMode !== oldMode) {
        this.clearResults();
        // 不自动重新检测，让用户手动重新上传或检测
      }
    }
  },
  created: function () {
    document.title = "海洋鱼类识别系统";
  },
  methods: {
    
    true_upload() {
      // 清空所有状态
      this.url_1 = "";
      this.url_2 = "";
      this.srcList1 = [];
      this.srcList2 = [];
      this.fishCount = 0;
      this.detections = [];
      this.yucejieguo = "";
      this.loading = false;
      this.dialogTableVisible = false;
      
      // 重置文件输入
      this.$refs.upload.value = '';
      this.$refs.upload.click();
    },
    true_upload2() {
      this.$refs.upload2.click();
    },
    
    // 清空所有内容（包括原图和检测结果）
    clearResults() {
      this.url_1 = "";
      this.url_2 = "";
      this.srcList1 = [];
      this.srcList2 = [];
      this.fishCount = 0;
      this.detections = [];
      this.yucejieguo = "";
      this.loading = false;
      this.dialogTableVisible = false;
      
      // 重置文件输入
      if (this.$refs.upload) {
        this.$refs.upload.value = '';
      }
      
      // 强制刷新图片显示
      this.$forceUpdate();
    },
    
    // 重新检测当前图片
    async reDetect() {
      if (!this.url_1) {
        this.$message.warning('请先上传图片');
        return;
      }
      
      // 从当前显示的URL获取文件名
      let imageName = this.url_1.split('/').pop().split('?')[0]; // 去除时间戳参数
      
      try {
        this.loading = true;
        this.dialogTableVisible = true;
        this.percentage = 0;
        
        var timer = setInterval(() => {
          this.myFunc();
        }, 30);
        
        // 根据检测模式选择API端点
        let endpoint = this.detectionMode === "multi" ? "/multi_detect" : "/upload";
        
        // 构建请求 - 使用已存在的图片文件名
        const response = await axios.post(
          `${this.server_url}${endpoint}?redetect=${imageName}`,
          {},
          { headers: { "Content-Type": "application/json" } }
        );
        
        this.percentage = 100;
        clearInterval(timer);          // 处理检测结果
        if (this.detectionMode === "multi") {
          // 添加时间戳防止缓存
          this.url_2 = response.data.detection_url + '?t=' + new Date().getTime();
          this.fishCount = response.data.fish_count || 0;
          // 直接使用后端返回的英文名称
          this.detections = response.data.detections || [];
          this.yucejieguo = `检测到 ${this.fishCount} 条鱼`;
        } else {
          // 添加时间戳防止缓存
          this.url_2 = response.data.draw_url + '?t=' + new Date().getTime();
          // 直接使用后端返回的英文名称
          let fishName = response.data.fish_name || response.data.yucejieguo;
          
          this.yucejieguo = fishName;
          this.fishCount = 1;
          this.detections = [{
            class: fishName,
            confidence: response.data.confidence || 0.85
          }];
        }
        
        this.srcList2.push(this.url_2);
        this.loading = false;
        this.dialogTableVisible = false;
        this.percentage = 0;
        
        // 强制更新图像显示
        this.$nextTick(() => {
          this.$forceUpdate();
        });
        
        this.notice();
        
      } catch (error) {
        console.error('重新检测失败:', error);
        this.loading = false;
        this.dialogTableVisible = false;
        this.$message.error('检测失败，请重新上传图片');
      }
    },
    
    // 获得目标文件
    getObjectURL(file) {
      var url = null;
      if (window.createObjcectURL != undefined) {
        url = window.createOjcectURL(file);
      } else if (window.URL != undefined) {
        url = window.URL.createObjectURL(file);
      } else if (window.webkitURL != undefined) {
        url = window.webkitURL.createObjectURL(file);
      }
      return url;
    },
  
    // 上传文件
    update(e) {
      this.percentage = 0;
      this.dialogTableVisible = true;
      this.url_1 = "";
      this.url_2 = "";
      this.srcList1 = [];
      this.srcList2 = [];
      this.wait_return = "";
      this.wait_upload = "";
      this.fullscreenLoading = true;
      this.loading = true;
      this.showbutton = false;
      let file = e.target.files[0];
      this.url_1 = this.$options.methods.getObjectURL(file);
      let param = new FormData(); //创建form对象
      param.append("file", file, file.name); //通过append向form对象添加数据
      //console.log(param.get("file")); //FormData私有类对象，访问不到，可以通过get判断值是否传进去
      var timer = setInterval(() => {
        this.myFunc();
      }, 30);
      let config = {
        headers: { "Content-Type": "multipart/form-data" },
      }; //添加请求头
      
      // 根据检测模式选择API端点
      let endpoint = this.detectionMode === "multi" ? "/multi_detect" : "/upload";
      
      axios
        .post(this.server_url + endpoint, param, config)
        .then((response) => {
          this.percentage = 100;
          clearInterval(timer);
          this.url_1 = response.data.image_url;
          this.srcList1.push(this.url_1);
          
          // 处理检测结果URL
          if (this.detectionMode === "multi") {
            this.url_2 = response.data.detection_url + '?t=' + new Date().getTime();
            this.fishCount = response.data.fish_count || 0;
            // 直接使用后端返回的英文名称
            this.detections = response.data.detections || [];
            this.yucejieguo = `检测到 ${this.fishCount} 条鱼`;
          } else {
            this.url_2 = response.data.draw_url + '?t=' + new Date().getTime();
            // 直接使用后端返回的英文名称
            let fishName = response.data.fish_name || response.data.yucejieguo;
            
            this.yucejieguo = fishName;
            this.fishCount = 1;
            // 为单鱼检测创建检测结果格式
            this.detections = [{
              class: fishName,
              confidence: response.data.confidence || 0.85
            }];
          }
          
          this.srcList2.push(this.url_2);
          this.fullscreenLoading = false;
          this.loading = false;
          this.dialogTableVisible = false;
          this.percentage = 0;
          this.notice();
        });
      },
    myFunc() {
      if (this.percentage + 33 < 99) {
        this.percentage = this.percentage + 33;
        this.percentage;
      } else {
        this.percentage = 99;
      }
    },
    notice() {
      let message = "点击预测图像可查看大图";
      let title = "预测成功";
      
      if (this.detectionMode === "multi" && this.fishCount > 0) {
        title = "多鱼检测成功";
        message = `检测到 ${this.fishCount} 条鱼，点击图像查看标注结果`;
      }
      
      this.$notify({
        title: title,
        message: message,
        duration: 0,
        type: "success",
      })
    }
  }
}
</script>

<style>
.el-button {
  padding: 12px 20px !important;
}

#hello p {
  font-size: 15px !important;
  /*line-height: 25px;*/
}

.n1 .el-step__description {
  padding-right: 20%;
  font-size: 14px;
  line-height: 20px;
  /* font-weight: 400; */
}
</style>

<style scoped>
/* 全局样式重置 */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

/* 主容器样式 */
#Content {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  padding: 20px;
  font-family: 'Helvetica Neue', Arial, sans-serif;
}

.main-container {
  display: flex;
  gap: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

/* 侧边栏样式 */
.sidebar {
  width: 300px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.info-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
}

.info-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.card-header i {
  font-size: 18px;
  color: #3498db;
}

/* AI模型信息样式 */
.model-stats {
  padding: 15px 0;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
}

.stat-item:last-child {
  border-bottom: none;
}

.stat-item .label {
  font-size: 14px;
  color: #7f8c8d;
}

.stat-item .value {
  font-size: 14px;
  font-weight: 600;
  color: #2c3e50;
}

.stat-item .highlight {
  color: #27ae60;
  background: rgba(39, 174, 96, 0.1);
  padding: 2px 8px;
  border-radius: 12px;
}

/* 检测结果样式 */
.detection-results {
  padding: 15px 0;
}

.result-summary {
  text-align: center;
  margin-bottom: 20px;
}

.fish-count {
  font-size: 36px;
  font-weight: bold;
  color: #e74c3c;
  display: block;
}

.count-label {
  font-size: 14px;
  color: #7f8c8d;
  margin-top: 5px;
}

.detection-list {
  space-y: 10px;
}

.detection-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: rgba(52, 152, 219, 0.05);
  border-radius: 8px;
  border-left: 4px solid #3498db;
  margin-bottom: 8px;
}

.detection-index {
  width: 24px;
  height: 24px;
  background: #3498db;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  font-weight: bold;
}

.detection-info {
  flex: 1;
}

.fish-name {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 2px;
}

.confidence {
  font-size: 12px;
  color: #27ae60;
}

.more-results {
  text-align: center;
  color: #7f8c8d;
  font-size: 12px;
  margin-top: 10px;
}

/* 单鱼检测结果样式 */
.single-fish-result {
  padding: 15px 0;
}

.fish-card {
  display: flex;
  align-items: center;
  gap: 15px;
  padding: 15px;
  background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
  border-radius: 12px;
  color: white;
  box-shadow: 0 4px 15px rgba(52, 152, 219, 0.3);
}

.fish-icon {
  font-size: 32px;
  animation: swim 2s ease-in-out infinite alternate;
}

@keyframes swim {
  from { transform: translateX(0) rotate(0deg); }
  to { transform: translateX(5px) rotate(2deg); }
}

.fish-details {
  flex: 1;
}

.fish-details .fish-name {
  font-size: 18px;
  font-weight: bold;
  margin-bottom: 5px;
}

.fish-details .confidence {
  font-size: 14px;
  opacity: 0.9;
}

.no-result {
  text-align: center;
  padding: 30px 0;
  color: #bdc3c7;
}

.no-result i {
  font-size: 48px;
  margin-bottom: 15px;
  display: block;
}

.no-result p {
  margin: 5px 0;
}

.no-result .tip {
  font-size: 12px;
  color: #95a5a6;
}

/* 主内容区域样式 */
.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* 控制面板样式 */
.control-panel {
  width: 100%;
}

.control-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.mode-selector {
  padding: 20px;
  text-align: center;
}

.section-title {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 18px;
  color: #2c3e50;
  margin-bottom: 20px;
  font-weight: 600;
}

.mode-buttons {
  display: flex;
  justify-content: center;
  gap: 10px;
}

.mode-btn {
  display: flex;
  align-items: center;
  gap: 5px;
}

/* 图片展示区域样式 */
.image-display {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}

.image-section {
  display: flex;
  flex-direction: column;
}

.image-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  height: 550px;
  transition: all 0.3s ease;
}

.image-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 15px 30px rgba(0, 0, 0, 0.1);
}

.image-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 16px;
  font-weight: 600;
  color: #2c3e50;
}

.image-header i {
  color: #9b59b6;
}

.image-content {
  height: calc(100% - 60px);
  display: flex;
  align-items: center;
  justify-content: center;
}

.image-wrapper {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.display-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 上传区域样式 */
.upload-area {
  width: 100%;
  height: 100%;
  border: 2px dashed #bdc3c7;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: rgba(236, 240, 241, 0.3);
}

.upload-area:hover {
  border-color: #3498db;
  background: rgba(52, 152, 219, 0.1);
}

.upload-content {
  text-align: center;
  padding: 40px 20px;
}

.upload-icon {
  font-size: 48px;
  color: #bdc3c7;
  margin-bottom: 15px;
  display: block;
}

.upload-text {
  font-size: 16px;
  color: #2c3e50;
  margin-bottom: 8px;
}

.upload-hint {
  font-size: 12px;
  color: #7f8c8d;
}

/* 等待区域样式 */
.waiting-area {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(236, 240, 241, 0.3);
  border-radius: 12px;
}

.waiting-content {
  text-align: center;
  padding: 40px 20px;
}

.waiting-icon {
  font-size: 48px;
  color: #3498db;
  margin-bottom: 15px;
  display: block;
}

.waiting-icon.el-icon-loading {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.waiting-text {
  font-size: 14px;
  color: #7f8c8d;
}

/* 操作按钮样式 */
.action-panel {
  display: flex;
  justify-content: center;
  gap: 15px;
  margin-top: 20px;
}

.action-button {
  padding: 12px 30px;
  font-size: 16px;
  border-radius: 25px;
  border: none;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
}

.action-button.detect-button {
  background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
}

.action-button:not(.detect-button) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.action-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
}

.action-button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}

/* 对话框样式优化 */
.el-dialog__header {
  text-align: center;
  padding-bottom: 10px;
}

.el-dialog__body {
  padding: 20px 30px 30px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .main-container {
    flex-direction: column;
  }
  
  .sidebar {
    width: 100%;
    flex-direction: row;
  }
  
  .image-display {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .sidebar {
    flex-direction: column;
  }
  
  .mode-buttons {
    flex-direction: column;
  }
}

</style>


