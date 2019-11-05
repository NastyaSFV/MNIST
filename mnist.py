from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K

# количество обучающих образцов, обрабатываемых одновременно за одну итерацию алгоритма градиентного спуска
batch_size = 128
# количество классов (по одному на цифру)
num_classes = 10
# количество итераций обучающего алгоритма по всему обучающему множеству
epochs = 12

# размер картинки
img_rows, img_cols = 28, 28

# подгрузка данных из файла
train = pd.read_csv("input/mnist_train.csv")
Y = train['label']
X = train.drop(['label'], axis=1)

# разделение на обучающую выборку и выборку валидации(10%)
x_train, x_val, y_train, y_val = train_test_split(X.values, Y.values, test_size=0.10)

# вывод на экран данных, соответствующих каждой цифре в обучающей выборке
l1 = list(y_train)
print("кол-во данных в обучающей выборке ")
print("кол-во 0:", l1.count(0))
print("кол-во 1:", l1.count(1))
print("кол-во 2:", l1.count(2))
print("кол-во 3:", l1.count(3))
print("кол-во 4:", l1.count(4))
print("кол-во 5:", l1.count(5))
print("кол-во 6:", l1.count(6))
print("кол-во 7:", l1.count(7))
print("кол-во 8:", l1.count(8))
print("кол-во 9:", l1.count(9))

# формат картинки 1D
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
# нормализация данных в диапазон [0, 1]
x_train /= 255
x_val /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'test samples')

# кодирование по label
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

# определение модели
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# скрытый слой relu
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
# наружный слой softmax
model.add(Dense(10, activation='softmax'))

# Был optimizer Adadelta, поэтому был низкий процент успеха
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# тренировка модели
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

# оценка тренировки модели на валидационной выборке
accuracy = model.evaluate(x_val, y_val, verbose=0)

# сохранение весов
model.save("output/mnist_weights.h5")