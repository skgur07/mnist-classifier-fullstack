<template>
  <div class="app-container">
    <div class="clean-card">
      
      <div class="chart-section">
        <h2 class="title">AI 숫자 예측</h2>
        
        <div class="prob-list">
          <div v-for="(prob, index) in probabilities" :key="index" class="prob-row">
            <span class="number-label">{{ index }}</span>
            <div class="bar-background">
              <div 
                class="bar-fill" 
                :style="{ width: (prob * 100) + '%', backgroundColor: getBarColor(prob, index) }"
              ></div>
            </div>
            <span class="percent-label">{{ (prob * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>

      <div class="canvas-section">
        <h2 class="title">숫자를 그려보세요</h2>
        
        <div class="canvas-wrapper">
          <canvas 
            ref="canvasRef" width="28" height="28" 
            @mousedown="startDrawing" @mousemove="draw" 
            @mouseup="handleMouseUp" @mouseleave="handleMouseUp"
            :class="{ 'waiting': isWaiting, 'clearing': isAutoClearing }"
          ></canvas>
        </div>

        <div class="status-text">
          <span v-if="isWaiting" class="text-orange">추가 획을 기다리는 중...</span>
          <span v-else-if="isAutoClearing" class="text-blue">잠시 후 지워집니다</span>
          <span v-else class="text-gray">여러 획을 이어서 그릴 수 있습니다</span>
        </div>

        <div class="result-box">
          결과: <span class="result-number">{{ result !== null ? result : '?' }}</span>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const canvasRef = ref(null);
const isDrawing = ref(false);
const result = ref(null);
const probabilities = ref(new Array(10).fill(0));

const isWaiting = ref(false);
const isAutoClearing = ref(false);

let ctx = null;
let completeTimer = null; 
let clearTimer = null;    

onMounted(() => {
  ctx = canvasRef.value.getContext('2d', { willReadFrequently: true });
  resetCanvas();
});

const resetCanvas = () => {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, 28, 28);
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2.2;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  isAutoClearing.value = false;
  isWaiting.value = false;
};

const startDrawing = (e) => {
  clearTimeout(completeTimer);
  clearTimeout(clearTimer);
  
  if (isAutoClearing.value) {
    resetCanvas();
    result.value = null;
    probabilities.value = new Array(10).fill(0);
  }
  
  isWaiting.value = false;
  isDrawing.value = true;
  draw(e);
};

const draw = (e) => {
  if (!isDrawing.value) return;
  const rect = canvasRef.value.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvasRef.value.width / rect.width);
  const y = (e.clientY - rect.top) * (canvasRef.value.height / rect.height);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y);
};

const handleMouseUp = () => {
  if (isDrawing.value) {
    isDrawing.value = false;
    ctx.beginPath();

    predict(); 

    isWaiting.value = true;
    completeTimer = setTimeout(() => {
      isWaiting.value = false;
      isAutoClearing.value = true;
      
      clearTimer = setTimeout(() => {
        resetCanvas();
        result.value = null;
        probabilities.value = new Array(10).fill(0);
      }, 2000); 
    }, 800); 
  }
};

const predict = async () => {
  const imageData = canvasRef.value.toDataURL('image/png');
  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    const data = await response.json();
    result.value = data.prediction;
    probabilities.value = data.probabilities;
  } catch (error) { 
    console.error("서버 연결 실패"); 
  }
};

const getBarColor = (prob, index) => {
  if (index === result.value && prob > 0.1) return '#3b82f6';
  return prob > 0.5 ? '#10b981' : '#e2e8f0';
};
</script>

<style src="./MnistCanvas.css" scoped></style>