import keras
from keras import layers

# mnist 데이터셋 가져오기
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# in-place 연산 때문에 사용하면 안됨
# x_train /= 255.0
# x_test /= 255.0

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., None]
x_test = x_test[..., None]

model = keras.models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPool2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"{test_accuracy * 100:.2f}%")

# 요즘 .h5을 사용하지 않음 
model.save("mnist_cnn_model.keras")
