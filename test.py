import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
img_rows, img_cols = 28, 28

# подгрузка данных из файлаи
test = pd.read_csv("input/mnist_test.csv")
Y = test['label']
X = test.drop(['label'], axis=1)
y_test = Y.values
x_test = X.values
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_test = x_test.astype('float32')
x_test /= 255

# подгрузка весов
model = keras.models.load_model("output/mnist_weights.h5")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# запуск нс
predict = model.predict(x_test)
predict = list(map(np.argmax, predict))

# создание матрицы
con_mat = tf.math.confusion_matrix(labels=y_test, predictions=predict).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)
# print(con_mat_df)

# отрисовка матрицы
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Reds)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()