import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def tran_and_save_model(path_to_save):
    batch_size = 256
    num_classes = 10
    epochs = 10

    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 处理 x
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    # 处理 y
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)

    inputs = Input(shape=(28, 28, 1))
    intermdia_layer = Conv2D(6, (5, 5), activation = 'relu', padding = 'same', input_shape = input_shape)(inputs)
    intermdia_layer = MaxPooling2D(pool_size = (2, 2))(intermdia_layer)

    intermdia_layer = Conv2D(16, (5, 5), activation = 'relu', padding = 'same')(intermdia_layer)
    intermdia_layer = MaxPooling2D(pool_size = (2, 2))(intermdia_layer)

    intermdia_layer = Flatten()(intermdia_layer)
    intermdia_layer = Dense(84, activation = 'relu')(intermdia_layer)
    predictions = Dense(10, activation = 'softmax')(intermdia_layer)
    model = Model(inputs=inputs, outputs=predictions)

    '''
    model = Sequential()

    model.add(Conv2D(6, (5, 5), activation = 'relu', padding = 'same', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(16, (5, 5), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(84, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    '''

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adadelta', metrics = ['accuracy'])

    model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = batch_size, epochs = epochs, verbose = 1)

    score = model.evaluate(x_test, y_test, verbose = 0)
    model.save('./models/myletnet4')
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

model_path='./models/myletnet4'
tran_and_save_model(model_path)

from keras.models import load_model
lenet4_model=load_model(model_path)
lenet4_model.summary()