from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
import pylab


def get_output_from_layers(sqmodel, start_index, end_index, x):
    # ret_dict=dict{image_index:{layername:feature}}
    ret_dict = {}
    for layer_index in range(start_index, end_index):
        # print(sqmodel.layers[i].output)
        # print(sqmodel.layers[i].output.name)
        layer_model = Model(inputs=sqmodel.input, outputs=sqmodel.layers[layer_index].output)
        features = layer_model.predict(x)  # [images,长,宽,features]
        # weights = sqmodel.layers[i].get_weights()
        # print(i)
        # print(\"sqmodel.layers[i].output.name:\")
        # print(sqmodel.layers[i].output.name)
        # print(\"features.shape:\")
        # print(features.shape)
        # for weight in weights:
        # print(\"weight.shape:\")
        # print(weight.shape)
        # print(\"-----------------------------------------------------------------------------------------\")
        cur_layer_nm = sqmodel.layers[layer_index].output.name
        for image_index in range(0, len(features)):
            # print(cur_layer_nm)
            if (image_index in ret_dict):
                ret_dict[image_index][cur_layer_nm] = features[image_index]
            else:
                cur_image_layer_dict = {cur_layer_nm: features[image_index]}
                ret_dict[image_index] = cur_image_layer_dict
    return ret_dict


def get_output_dict_from_layers(sqmodel, start_index, end_index, x):
    # ret_dict=dict{image_index:{layername:feature}}
    ret_dict = {}
    for layer_index in range(start_index, end_index):
        layer_model = Model(inputs=sqmodel.input, outputs=sqmodel.layers[layer_index].output)
        features = layer_model.predict(x)  # [images,长,宽,features]
        cur_layer_nm = sqmodel.layers[layer_index].output.name
        for image_index in range(0, len(features)):
            # print(cur_layer_nm)
            if (image_index in ret_dict):
                ret_dict[image_index][cur_layer_nm] = features[image_index]
            else:
                cur_image_layer_dict = {cur_layer_nm: features[image_index]}
                ret_dict[image_index] = cur_image_layer_dict
    return ret_dict


def get_output_numpy_from_layers(sqmodel, start_index, end_index, x):
    ret_list = []
    for layer_index in range(start_index, end_index):
        layer_model = Model(inputs=sqmodel.input, outputs=sqmodel.layers[layer_index].output)
        features = layer_model.predict(x)  # [images,长,宽,features]
        feature_index = 0
        for image_index in range(0, len(features)):
            if (image_index < len(ret_list)):
                ret_list[image_index].append(features[image_index])
            else:
                ret_list.append([features[image_index]])
        feature_index += 1
    return np.array(ret_list)


def output_fully_connected_features_to_graph(features_dict):
    for image in features_dict:
        print("t # " + str(image))
        # for layer_index in range(0,len(features_dict[image])-1):
        for layer_name in features_dict[image]:
            print(layer_name)
        print(features_dict[image][layer_name].shape)
        # cur_layer_out = features_dict[image][cur_layer_name]
        # next_layer_name = features_dict[image][layer_index+1]
        # next_layer_out = features_dict[image][next_layer_name]
        # print(cur_layer_out.shape)
        # print(next_layer_out.shape)
        print("-----------------------------------------------------------------------------------------")
        # for cur_layer_feature in
        # print(\"v \" + layer_index + layer_out)
        # print(layer_out)
        # print(features[image][layer_out].shape)


def output_fully_connected_features_to_graph(features_numpy, delta_layer_index=1):
    for image_index in range(0, len(features_numpy)):
        graph_label_str = str("t # " + str(image_index))
        # print(graph_label_str)
        write_to_file("./outputs/test-output-graph.txt", graph_label_str)
        graph_v_index = 0
        image_features = features_numpy[image_index]
        # output vertex info
        for layer_index in range(0, len(image_features)):
            cur_layer_features_cnt = image_features[layer_index].shape[-1]
            for cur_layer_feature_index in range(0, cur_layer_features_cnt):
                v_info_str = str("v " + str(graph_v_index) + " " + get_v_label(layer_index + delta_layer_index,
                                                                               cur_layer_feature_index))
                # print(v_info_str)
                write_to_file("./outputs/test-output-graph.txt", v_info_str)
                graph_v_index += 1

        # output edge info
        for layer_index in range(0, len(image_features)):
            if layer_index < len(image_features) - 1:
                cur_layer_features_cnt = image_features[layer_index].shape[-1]
                next_layer_features_cnt = image_features[layer_index + 1].shape[-1]
                for cur_layer_feature_index in range(0, cur_layer_features_cnt):
                    cur_layer_feature_maps = image_features[layer_index]
                    cur_layer_feature_map = get_tensor_by_last_axis(cur_layer_feature_maps, cur_layer_feature_index)
                    # if():cur_layer_feature_map,待补充
                    for next_layer_feature_index in range(0, next_layer_features_cnt):
                        next_layer_feature_maps = image_features[layer_index + 1]
                        next_layer_feature_map = get_tensor_by_last_axis(next_layer_feature_maps,
                                                                         next_layer_feature_index)
                        e_info_str = str("e " + str(cur_layer_feature_index) + " " + str(
                            next_layer_feature_index + cur_layer_features_cnt) + " " + get_v_label(
                            layer_index + delta_layer_index, cur_layer_feature_index) + "-" + get_v_label(
                            layer_index + delta_layer_index + 1, next_layer_feature_index))
                        # print(e_info_str)
                        write_to_file("./outputs/test-output-graph.txt", e_info_str)


def get_v_label(layer_index, cur_layer_feature_index):
    return 'ly_' + str(layer_index) + "_" + str(cur_layer_feature_index)


def get_tensor_by_last_axis(feature, index_in_last_axis):
    ndim = feature.ndim
    if (ndim == 1):
        return feature[index_in_last_axis]
    elif (ndim == 2):
        return feature[:, index_in_last_axis]
    elif (ndim == 3):
        return feature[:, :, index_in_last_axis]
    elif (ndim == 4):
        return feature[:, :, :, index_in_last_axis]
    elif (ndim == 5):
        return feature[:, :, :, :, index_in_last_axis]
    elif (ndim == 6):
        return feature[:, :, :, :, :, index_in_last_axis]
    elif (ndim == 7):
        return feature[:, :, :, :, :, :, index_in_last_axis]
    elif (ndim == 8):
        return feature[:, :, :, :, :, :, :, index_in_last_axis]
    else:
        print("the ndim of tensor must be <= 8")
        return;


def write_to_file(path, str):
    with open(path, 'a') as f:
        f.write(str + "\n")


# print single image's feature-maps
def print_feature_map_infos(features):
    print("feature-map infos:" + str(features.shape))
    for i in range(0, features.shape[-1]):
        print("feature-map " + str(i) + " ----------------------------------------------------------")
        print(get_tensor_by_last_axis(features, i))
        #if(np.sum(get_tensor_by_last_axis(features, i))==0):
        #    print("feature-map " + str(i) + " ----------------------------------------------------------")
        #    print(get_tensor_by_last_axis(features, i))
def convert_tensor_2_scalar(tensor, op):
    if(op =='sum'):
        return np.sum(tensor)
    elif(op=='mean'):
        return  np.mean(tensor)
    elif(op=='median'):
        return np.median(tensor)
    elif(op=='amax'):
        return np.amax(tensor)
    elif(op=='amin'):
        return np.amin(tensor)
    elif(op=='var'):
        return np.var(tensor)
    else:
        print("error")


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

for image in features:
    print_feature_map_infos(image)

if (features[0].ndim == 3):
    pylab.imshow(features[0][:, :, 0])
    pylab.show()

layers_len = len(base_model.layers)
# out_put=get_output_from_layers(base_model,1,layers_num,x)
# output_fully_connected_features_to_graph(out_put)

# out_put = get_output_numpy_from_layers(base_model, 21, layers_len, x)
# output_fully_connected_features_to_graph(out_put, 21)
