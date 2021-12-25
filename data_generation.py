import tensorflow as tf
import os
import numpy as np
from PIL import Image
import functions_new



root = '/root/JsrNet 2.0/'
root_path = root + 'test_1_rearranged'
data_class = 'validation'
folder_path = os.listdir(root_path + '/' + data_class)
num_height = 160
num_width = 160
num_output = 4

if num_output == 2:
    output_filename = root + 'TFdata' + '/' + data_class + '.tfrecord'
elif num_output == 1:
    output_filename = root + 'TFdata' + '/' + data_class + '.tfrecord'
else:
    if num_output == 4:
        output_filename = root + 'TFdata' + '/'  + data_class + '.tfrecord'



count_batch = 0


if num_output == 2:
    with tf.io.TFRecordWriter(output_filename) as writer:
        for i in range(len(folder_path)):
            folder_path_unsorted = os.listdir(root_path + '/'+ data_class +'/' + folder_path[i])
            folder_path_unsorted.sort()
            folder_path_sorted = folder_path_unsorted
            key_1 = Image.open(root_path + '/'+ data_class + '/' + folder_path[i] + '/' + folder_path_sorted[0])
            key_1 = np.asarray(key_1, np.float32)
            key_1 = key_1 / 255
            reference_1 = key_1.astype(np.float32)


            for j in range(1,len(folder_path_sorted)):
                non_key = Image.open(root_path + '/' + data_class+ '/' + folder_path[i] + '/' + folder_path_sorted[j])
                non_key = np.asarray(non_key, np.float32)
                non_key = non_key / 255
                non_key = non_key.astype(np.float32)
                data = np.zeros((num_height, num_width, num_output), dtype=np.float32)
                data[:,:,0] = reference_1
                data[:,:,1] = non_key
                #################################################3
                # label = data[:,:,1]
                # data = data.reshape(num_height * num_width * 2, 1)
                # label = label.reshape(num_height * num_width , 1)
                ##################################################3
                data = data.reshape(num_height * num_width * num_output, 1)
                label = data
                ######################################################################
                data = data.astype(np.float32)
                binary_data = data.tobytes()##将图片转化为bites
                label = label.astype(np.float32)
                binary_label = label.tobytes()
                tf_example = functions_new.data_example(binary_data, binary_label)
                count_batch = count_batch + 1
                writer.write(tf_example.SerializeToString())
                print(count_batch)

####################################################################################################
if num_output == 4:
    with tf.io.TFRecordWriter(output_filename) as writer:
        for i in range(len(folder_path)):
            folder_path_unsorted = os.listdir(root_path + '/'+ data_class +'/' + folder_path[i])
            folder_path_unsorted.sort()
            folder_path_sorted = folder_path_unsorted
            key_1 = Image.open(root_path + '/'+ data_class + '/' + folder_path[i] + '/' + folder_path_sorted[0])
            key_1 = np.asarray(key_1, np.float32)
            key_1 = key_1 / 255
            reference_1 = key_1.astype(np.float32)
            count = 1
            data = np.zeros((num_height, num_width, num_output), dtype=np.float32)
            for j in range(1,len(folder_path_sorted)):
                non_key = Image.open(root_path + '/' + data_class+ '/' + folder_path[i] + '/' + folder_path_sorted[j])
                non_key = np.asarray(non_key, np.float32)
                non_key = non_key / 255
                non_key = non_key.astype(np.float32)
                data[:,:,0] = reference_1
                data[:,:,count] = non_key
                count = count + 1
                #################################################3
                # label = data[:,:,1]
                # data = data.reshape(num_height * num_width * 2, 1)
                # label = label.reshape(num_height * num_width , 1)
                ##################################################3
            data = data.reshape(num_height * num_width * num_output, 1)
            label = data
                ######################################################################
            data = data.astype(np.float32)
            binary_data = data.tobytes()
            label = label.astype(np.float32)
            binary_label = label.tobytes()
            tf_example = functions_new.data_example(binary_data, binary_label)
            count_batch = count_batch + 1
            writer.write(tf_example.SerializeToString())
            print(count_batch)

print("lenth: ", len(folder_path))

a = 1
