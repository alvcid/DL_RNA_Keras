import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers

# Download the data
mnist = datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print(X_train.shape)

# Visualize the data
# plt.figure(figsize=(20, 4))
#
# for index, digit in zip(range(1,9), X_train[:8]):
#     plt.subplot(1, 8, index)
#     plt.imshow(np.reshape(digit, (28, 28)), cmap=plt.cm.gray)
#     plt.title("Ejemplo: " + str(index))
# plt.show()

# Divide the data
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# NN architecture
network = models.Sequential()

network.add(layers.Dense(300, activation="relu", input_shape=(28*28,)))
network.add(layers.Dense(100, activation="relu"))
network.add(layers.Dense(10, activation="softmax"))

network.compile(loss="categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy", "Precision"])

print(network.summary())

# Prepare data
# Prepare data for input shape
X_train_prep = X_train.reshape((60000, 28*28))
X_train_prep = X_train_prep.astype('float32') / 255

X_test_prep = X_test.reshape((5000, 28*28))
X_test_prep = X_test_prep.astype('float32') / 255

X_val_prep = X_val.reshape((5000, 28*28))
X_val_prep = X_val_prep.astype('float32') / 255

# Prepare data for output shape
y_train_prep = to_categorical(y_train)
y_test_prep = to_categorical(y_test)
y_val_prep = to_categorical(y_val)

# Train the model
history = network.fit(X_train_prep,
                      y_train_prep,
                      epochs=10,
                      validation_data=(X_val_prep, y_val_prep))

pd.DataFrame(history.history).plot(figsize=(10,7))
plt.grid(True)
plt.gca().set_ylim(0, 1.2)
plt.xlabel("epochs")
plt.show()

# Validate the dataset
test_loss, test_acc, test_prec = network.evaluate(X_test_prep, y_test_prep)

print("test_acc: ", test_acc)
print("test_prec: ", test_prec)

# Save the model
# network.save("model_mnist.h5")
