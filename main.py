import numpy as np # пакет для работы с векторами и матрицами
import matplotlib.pyplot as plt #функция для отображения графиков
from tensorflow.keras.datasets import mnist  # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input #слои, кот мы будем использовать

#стандартизация входных данных
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255
x_test = x_test/255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

batch_size = 100
input_img = Input((28,28,1)) #входы НС будут соотв этому изображению
x = Flatten()(input_img) #вытягиваем изображение в один вектор
x = Dense(128, activation='relu')(x) #подаем этот вектор на полносвязный слой НС
x = Dense(64, activation='relu')(x) # следующий полносвязный слой
encoded = Dense(49, activation='relu')(x)#слой скрытого состояния

d = Dense(128, activation='relu')(encoded)
d = Dense(28 * 28, activation='sigmoid')(d)
decoded = Reshape((28, 28, 1))(d) #восстановленное изображение на выходе

autoencoder = keras.Model(input_img, decoded, name="autoencoder") #формируем модедь автоэнкодера
autoencoder.compile(optimizer='adam', loss='mean_squared_error') #компилируем НС

#запускаем процесс обучения, и на вход, и на выход подаем один и тот же сигнал

autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=batch_size,
                shuffle=True) #перемешиваем наблюдения при обучении

#отображаем первые десять изображений и результат
n = 10

imgs = x_test[:n]
decoded_imgs = autoencoder.predict(x_test[:n], batch_size=n)

plt.figure(figsize=(n, 2))
for i in range(n):
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(imgs[i].squeeze(), cmap='gray')
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  ax2 = plt.subplot(2, n, i + n + 1)
  plt.imshow(decoded_imgs[i].squeeze(), cmap='gray')
  ax2.get_xaxis().set_visible(False)
  ax2.get_yaxis().set_visible(False)

plt.show()

def plot_digits(*images): #отображдаем в консоли цифры
  images = [x.squeeze() for x in images]
  n = images[0].shape[0]  # число изображений

  plt.figure(figsize=(n, len(images)))
  for j in range(n):
    for i in range(len(images)):
      ax = plt.subplot(len(images), n, i * n + j + 1)
      plt.imshow(images[i][j])
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)

  plt.show()
def plot_homotopy(frm, to, n=10, autoencoder=None):
  z = np.zeros(([n] + list(frm.shape)))
  for i, t in enumerate(np.linspace(0., 1., n)):
    z[i] = frm * (1 - t) + to * t  # Гомотопия по прямой
  if autoencoder:
    plot_digits(autoencoder.predict(z, batch_size=n))
  else:
    plot_digits(z)

frm, to = x_test[y_test == 5][1:3]
plot_homotopy(frm, to)
plot_homotopy(frm, to, autoencoder=autoencoder)