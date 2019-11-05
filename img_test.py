import numpy as np
from tensorflow import keras
import PIL
import glob
import matplotlib.pyplot as plt
import os

img_width, img_height = 28, 28
input_shape = (img_width, img_height, 1)
img_name = "test3.png"
data = []

# подгрузка веса
model = keras.models.load_model("output/mnist_weights.h5")

# переход в директорию проекта
os.chdir(os.getcwd() + "/img")

# обработка картинки
for file in glob.glob(img_name):
    img = PIL.Image.open(file).convert('L')
    # отрисовка картинки
    plt.imshow(img.convert('RGBA'))
    plt.show()
    img_array = np.array(img)
    for i in range(len(img_array)):
        for j in range(len(img_array[i])):
            img_array[i][j] = 255 - img_array[i][j]
    data.append(img_array)
    data = np.array(data)
data = data / 255
data = data.reshape(data.shape[0], 28, 28, 1)

# запуск нс
predict = model.predict(data)
prediction = np.argmax(predict)

print('С некоторой вероятностью, это число: ', prediction)