from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import pylab

base_model = VGG16(weights='imagenet')
base_model.summary()
# for layer in base_model.layers:
#    print(layer.output)
#    print(\"layer.input_shape:\")
#    print(layer.input_shape)
#
#    weights = layer.get_weights()
#    print(\"weight.shape:\")
#    for weight in weights:
#        print(weight.shape)
#
#    print(\"layer.output_shape:\")
#    print(layer.output_shape)
layer_model = Model(inputs=base_model.input, outputs=base_model.layers[18].output)
# print(len(base_model.layers))
# print(base_model.layers)
img_path = 'D:\\workspace\\JupyterNotebook\\keras learning\\imagenet.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print("origin image size: " + str(x.shape))
# write_to_file("./outputs/test-output.txt",str(x))
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
# print(len(x[0][0][0]))
features = layer_model.predict(x)

#for image in features:
#    print_feature_map_infos(image)

if (features[0].ndim == 3):
    pylab.imshow(features[0][:, :, 0])
    pylab.show()