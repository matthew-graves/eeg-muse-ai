import sys
import numpy
import h5py
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
numpy.set_printoptions(threshold=sys.maxsize, suppress=True)


def plot_value_array(i, predictions_array, true_label):
    # Function To Print Graph Displaying Probablillity
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def get_dataset(filename):
    # Load Dataset From File
    hf = h5py.File(filename, 'r')
    all_data = np.zeros([100, 5, 3])
    for i in hf:
        data = hf[i][:]
        all_data = np.concatenate((all_data, data), axis=0)

    all_data[:, :, 2] = (keras.utils.normalize(all_data[:, :, 2], order=2))
    return all_data


def separate_data(all_data):
    # Separate between Data & Labels
    labels = (all_data[:, :1, :1]).flatten()
    data_only = all_data[:, :, 1:]
    return data_only, labels


def create_model():
    # Create New Model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(5, 2)),
        keras.layers.Dense(25, activation=tf.nn.relu),
        keras.layers.Dense(25, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, data_only, labels, epochs):
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(data_only, labels, epochs=epochs, callbacks=callbacks_list, batch_size=100, validation_split=0.20,
              verbose=1)
    model.save('model.h5')
    return model


def plot_model(predictions_single, labels):
    plot_value_array(0, predictions_single, labels.astype(int))
    plt.xticks(range(3), ['red', 'white', 'blue'], rotation=45)
    plt.show()


def load_existing_model(filename):
    model = keras.models.load_model(filename)
    return model


def make_prediction(model, data_only, i):
    test = (np.expand_dims(data_only[i], 0))
    predictions_single = model.predict(test)
    prediction_result = np.argmax(predictions_single[0])
    print(int(prediction_result))


def make_random_prediction(model, data_only, labels):
    randnum = random.randint(100, 1100)
    test = (np.expand_dims(data_only[randnum], 0))
    predictions_single = model.predict(test)
    prediction_result = np.argmax(predictions_single[0])
    print(prediction_result, labels[randnum])


def default_action():
    all_data = get_dataset('data.h5')
    data_only, labels = separate_data(all_data)
    model = load_existing_model('model.h5')
    train_model(model, data_only, labels, 10000)
