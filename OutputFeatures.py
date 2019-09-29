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


# [image_index, layer_index, image_width, image_highth, channel_index]
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


def output_fully_connected_features_to_graph(features_numpy, delta_layer_index=1, func_list=[],
                                             graph_file="./garap-tmp.txt"):
    if len(features_numpy[0]) != len(func_list):
        print("error, illegal parameter func_list:" + str(func_list))
        return;
    with open(graph_file, 'w') as f:
        for image_index in range(0, len(features_numpy)):
            print("processing image: " + str(image_index))
            graph_label_str = str("t # " + str(image_index))
            # print(graph_label_str)
            f.write(graph_label_str + "\n")
            graph_v_index = 0
            image_features = features_numpy[image_index]
            # output vertex info
            for layer_index in range(0, len(image_features)):
                #print("processing image: "+str(image_index)+" layer: "+str(layer_index+delta_layer_index))
                cur_layer_features_cnt = image_features[layer_index].shape[-1]
                for cur_layer_feature_index in range(0, cur_layer_features_cnt):
                    v_info_str = str("v " + str(graph_v_index) + " " + get_v_label(layer_index + delta_layer_index,
                                                                                   cur_layer_feature_index))
                    # print(v_info_str)
                    f.write(v_info_str + "\n")
                    graph_v_index += 1

            # output edge info
            for layer_index in range(0, len(image_features)):
                if layer_index < len(image_features) - 1:
                    cur_layer_features_cnt = image_features[layer_index].shape[-1]
                    next_layer_features_cnt = image_features[layer_index + 1].shape[-1]

                    cur_layer_feature_maps = image_features[layer_index]
                    cur_layer_features_flags = func_list[layer_index](cur_layer_feature_maps)

                    for cur_layer_feature_index in range(0, cur_layer_features_cnt):
                        # If the user-defined threshold function:f is satisfied by f(feature), the f(feature) wiil be output.
                        cur_layer_feature_map = get_tensor_by_last_axis(cur_layer_feature_maps, cur_layer_feature_index)
                        if (cur_layer_features_flags[cur_layer_feature_index] > 0):
                            # cur_layer_feature_map to be use in the feature
                            for next_layer_feature_index in range(0, next_layer_features_cnt):
                                # next_layer_feature_maps = image_features[layer_index + 1]
                                # next_layer_feature_map = get_tensor_by_last_axis(next_layer_feature_maps, next_layer_feature_index)
                                e_info_str = str("e " + str(cur_layer_feature_index) + " " + str(
                                    next_layer_feature_index + cur_layer_features_cnt) + " " + get_v_label(
                                    layer_index + delta_layer_index, cur_layer_feature_index) + "-" + get_v_label(
                                    layer_index + delta_layer_index + 1, next_layer_feature_index))
                                # print(e_info_str)
                                f.write(e_info_str + "\n")
                        else:
                            e_info_str = str(get_v_label(layer_index + delta_layer_index,
                                                         cur_layer_feature_index) + " does not satisfy the output threshold function")
                            #print(e_info_str)
                            #print("cur_layer_feature_map: "+str(cur_layer_feature_map))
        f.write("t # -1\n")

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
        print("error, the ndim of tensor must be <= 8")
        return;


def write_to_file(path, str):
    with open(path, 'w') as f:
        f.write(str + "\n")


# print single image's feature-maps
def print_feature_map_infos(features):
    print("feature-map infos:" + str(features.shape))
    for i in range(0, features.shape[-1]):
        print("feature-map " + str(i) + " ----------------------------------------------------------")
        print(get_tensor_by_last_axis(features, i))
        # if(np.sum(get_tensor_by_last_axis(features, i))==0):
        #    print("feature-map " + str(i) + " ----------------------------------------------------------")
        #    print(get_tensor_by_last_axis(features, i))


def bigger_than_zero(feature_maps):
    return bigger_than_mean_t(feature_maps, 0)


def bigger_than_median(feature_maps):
    global_median = np.median(feature_maps)
    return bigger_than_median_t(feature_maps, global_median)


def bigger_than_median_t(feature_maps, t):
    fms_cnt = feature_maps.shape[-1]
    ret = np.zeros(fms_cnt, dtype=int)
    for i in range(0, fms_cnt):
        fm_median = np.median(get_tensor_by_last_axis(feature_maps, i))
        if (fm_median > t):
            ret[i] = 1
    return ret


def bigger_than_mean_val(feature_maps):
    global_mean = np.mean(feature_maps)
    return bigger_than_mean_t(feature_maps, global_mean)


def bigger_than_mean_t(feature_maps, t):
    fms_cnt = feature_maps.shape[-1]
    ret = np.zeros(fms_cnt, dtype=int)
    for i in range(0, fms_cnt):
        fm_mean = np.mean(get_tensor_by_last_axis(feature_maps, i))
        if (fm_mean > t):
            ret[i] = 1
    return ret


def get_argmax(feature_maps):
    fms_cnt = feature_maps.shape[-1]
    ret = np.zeros(fms_cnt, dtype=int)
    if (feature_maps.ndim == 1):
        ret[np.argmax(feature_maps)] = 1
        return ret
    return;


if __name__ == '__main__':
    base_model = VGG16(weights='imagenet')
    base_model.summary()
    layer_model = Model(inputs=base_model.input, outputs=base_model.layers[18].output)
    img_path = 'D:\\workspace\\JupyterNotebook\\keras learning\\imagenet.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = layer_model.predict(x)

    layers_len = len(base_model.layers)
    # out_put=get_output_from_layers(base_model,1,layers_num,x)
    # output_fully_connected_features_to_graph(out_put)

    out_put = get_output_numpy_from_layers(base_model, 16, 18, x)
    func_list = [bigger_than_zero, bigger_than_zero]
    graph_file = "./outputs/test-output-graph.txt"
    output_fully_connected_features_to_graph(out_put, 16, func_list, graph_file)
