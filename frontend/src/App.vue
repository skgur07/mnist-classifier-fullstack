<template>
  <MnistCanvas />
</template>

<script setup>
import MnistCanvas from './components/MnistCanvas.vue'
</script>

<style>
/* 브라우저 기본 여백 완전히 제거 */
body {
  margin: 0;
  padding: 0;
  background-color: #f1f5f9; /* 카드(흰색)가 돋보이도록 배경을 살짝 톤다운 */
}

#app {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
</style>