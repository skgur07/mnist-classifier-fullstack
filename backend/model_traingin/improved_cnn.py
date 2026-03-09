import keras
from keras import layers, models, callbacks

# 1. 데이터 로드 및 전처리
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

# 채널 차원 추가 (28, 28) -> (28, 28, 1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# 2. 향상된 모델 설계
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # 첫 번째 Conv 블록
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(), # 성능 향상의 핵심
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25), # 과적합 방지

    # 두 번째 Conv 블록
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # 완전 연결층 (Dense)
    layers.Flatten(),
    layers.Dense(256, activation='relu'), # 노드 수 증가
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 3. 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4. 콜백 함수 정의 (학습 효율 극대화)
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,          # 5번의 Epoch 동안 개선이 없으면 종료
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2,          # 손실이 안 줄어들면 학습률을 1/5로 낮춤
    patience=3
)

# 5. 학습 (Epoch를 20으로 늘림)
history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,      # 배치 사이즈 명시
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# 6. 평가 및 저장
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"최종 정확도: {test_accuracy * 100:.2f}%")

model.save("mnist_cnn_model.keras")