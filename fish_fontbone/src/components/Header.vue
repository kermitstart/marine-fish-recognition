<template>
  <div id="Header">
    <div class="header-content">
      <div class="logo-section">
        <i class="el-icon-view header-icon"></i>
        <h1 class="title">{{ msg }}</h1>
      </div>
      <div class="subtitle">
        <p>基于AI深度学习的智能海洋生物识别平台</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Header",
  data() {
    return {
      msg: "海洋鱼类识别系统",
      activeIndex: "1",
    };
  },
};
</script>

<style scoped>
#Header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 30px 0;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.header-content {
  max-width: 1400px;
  margin: 0 auto;
  text-align: center;
  padding: 0 20px;
}

.logo-section {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 15px;
  margin-bottom: 10px;
}

.header-icon {
  font-size: 48px;
  color: #ffffff;
  background: rgba(255, 255, 255, 0.2);
  padding: 15px;
  border-radius: 50%;
  backdrop-filter: blur(10px);
}

.title {
  font-size: 36px;
  font-weight: 700;
  color: #ffffff;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  letter-spacing: 3px;
  margin: 0;
}

.subtitle {
  margin-top: 10px;
}

.subtitle p {
  font-size: 16px;
  color: rgba(255, 255, 255, 0.9);
  margin: 0;
  font-weight: 300;
  letter-spacing: 1px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .logo-section {
    flex-direction: column;
    gap: 10px;
  }
  
  .header-icon {
    font-size: 36px;
    padding: 12px;
  }
  
  .title {
    font-size: 28px;
    letter-spacing: 2px;
  }
  
  .subtitle p {
    font-size: 14px;
  }
}

.el-menu-demo {
  width: 80%;
  margin: 0px auto;
  padding: 0px auto;
}

.top-left-edition span i {
  float: left;
  margin-right: 10px;
}

i,
input,
label {
  vertical-align: middle;
}

i {
  border: 0;
  display: block;
  cursor: pointer;
}

.top-left-edition span {
  float: left;
  font-size: 16px;
  color: #999999;
  line-height: 24px;
  margin-right: 40px;
}
</style>


