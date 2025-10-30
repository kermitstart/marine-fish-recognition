<template>
  <div id="Footer">
    <div class="footer-content">
      <div class="copyright">
        <p>{{ msg }}</p>
      </div>
      <div class="tech-info">
        <p>基于 PyTorch & Vue.js 构建 | AI准确率 91.62%</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "Footer",
  data() {
    return {
      msg: "© 2024 海洋鱼类识别系统 - 仅供学习研究使用",
    };
  },
};
</script>

<style scoped>
#Footer {
  background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
  padding: 25px 0;
  margin-top: 40px;
}

.footer-content {
  max-width: 1400px;
  margin: 0 auto;
  text-align: center;
  padding: 0 20px;
}

.copyright p {
  color: rgba(255, 255, 255, 0.9);
  font-size: 14px;
  margin: 0 0 8px 0;
  font-weight: 500;
}

.tech-info p {
  color: rgba(255, 255, 255, 0.7);
  font-size: 12px;
  margin: 0;
  font-weight: 300;
}
</style>


