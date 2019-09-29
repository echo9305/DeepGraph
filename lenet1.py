import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np


def get_mnist_data(img_rows=28, img_cols=28, channels=1):
    input_shape = (img_rows, img_cols, channels)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 处理 x
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    # 处理 y
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    return (x_train, y_train), (x_test, y_test)


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def tran_and_save_model(path_to_save):
    batch_size = 256
    num_classes = 10
    epochs = 10
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    inputs = Input(shape=(28, 28, 1))
    intermdia_layer = Conv2D(4, (5, 5), activation='relu', padding='same', input_shape=input_shape)(inputs)
    intermdia_layer = MaxPooling2D(pool_size=(2, 2))(intermdia_layer)

    intermdia_layer = Conv2D(12, (5, 5), activation='relu', padding='same')(intermdia_layer)
    intermdia_layer = MaxPooling2D(pool_size=(2, 2))(intermdia_layer)

    intermdia_layer = Flatten()(intermdia_layer)
    predictions = Dense(10, activation='softmax')(intermdia_layer)
    model = Model(inputs=inputs, outputs=predictions)

    '''
    model = Sequential()

    model.add(Conv2D(4, (5, 5), activation = 'relu', padding = 'same', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(12, (5, 5), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    '''

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, verbose=1)

    score = model.evaluate(x_test, y_test, verbose=2)
    model.save(path_to_save)


if __name__ == '__main__':
    model_path = './models/myletnet1'
    # tran_and_save_model(model_path)

    from keras.models import load_model

    lenet1_model = load_model(model_path)
    lenet1_model.summary()

    import OutputFeatures

    print("lanet layers len: " + str(len(lenet1_model.layers)))
    _, (x_test, y_test) = get_mnist_data()
    print(x_test.shape)
    out_put = OutputFeatures.get_output_numpy_from_layers(lenet1_model, 1, len(lenet1_model.layers), x_test)
    func_list = [OutputFeatures.bigger_than_zero, OutputFeatures.bigger_than_zero, OutputFeatures.bigger_than_zero,
                 OutputFeatures.bigger_than_zero, OutputFeatures.bigger_than_zero, OutputFeatures.get_argmax]
    OutputFeatures.output_fully_connected_features_to_graph(out_put, 1, func_list, "./outputs/garph_lenet1.txt")
