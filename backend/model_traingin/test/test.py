import numpy as np

from keras.models import load_model
from keras.utils import load_img, img_to_array

model_path = "model_traingin/mnist_cnn_model.keras"
model = load_model(model_path)

img_path = "model_traingin/image.png"
img = load_img(img_path, target_size=(28, 28), color_mode="grayscale")

img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

import pprint

pprint.pprint(f"예측 결과 배열 : {predictions[0]}")
print(f"최종 예측 숫자 : {predicted_class[0]}")