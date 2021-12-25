import tensorflow as tf
import functions
import os
from PIL import Image
import numpy as np
import math
import time
from functions import _psnr_3d
import matplotlib.pyplot as plt
from functions import _psnr





os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root = '/root/DINet/'
# filename_test = root + 'TFdata' + '/' + 'TFdata_3_output' + '/'+ 'test' + '.tfrecord'
#filename_test = root + 'TFdata' + '/' + 'TFdata_key_rec' + '/' + 'TFdata_1_output' + '/' + 'test' + '.tfrecord'
#filename_test = root + 'TFdata' + '/' + 'TFdata_original' + '/' + 'TFdata_1_output' + '/' + 'test' + '.tfrecord'
subrate_cs = 0.01
subrate_key = 0.25
weight_decay_rate = 0.01
num_output = 4
filename_test = root + 'TFdata' + '/' + 'TFdata_test_show' + '/' + 'TFdata_4_output' + '/' + 'test_11856' + '.tfrecord'
model_save_path = root + 'trained_model' + '/' + 'trained_model_4_output' + '/'+ str(subrate_cs)
####################
#model_save_path = root + 'trained_model' + '/' + trained_model_3_output' + '/' + str(subrate_cs)
# model_save_path = root + 'trained_model' + '/' + 'trained_model_1_output' + '/'+ str(subrate_cs)

model_name = 'DINet'  + '_' + str(subrate_cs)
block_size = 32

psnr_list_key = []
psnr_list_cs = []
ssim_list_key = []
ssim_list_cs = []
time_list = []
batch_size = 1

is_training = tf.placeholder(tf.bool)
handle = tf.placeholder(tf.string, shape=[])
count_test = functions.count_dataset(filename_test)
test_set = tf.data.TFRecordDataset(filename_test)
test_set = test_set.map(functions.parse_function)
#test_set = test_set.shuffle(count_test, reshuffle_each_iteration=True)
test_set = test_set.batch(batch_size,drop_remainder=False)
iterator = tf.data.Iterator.from_string_handle(handle, test_set.output_types, test_set.output_shapes)
data, label = iterator.get_next()
testing_iterator = test_set.make_initializable_iterator()

prediction = functions.DINet_4(data, block_size, subrate_key, subrate_cs, weight_decay_rate, is_training)
original = functions.collect_label(label)

saver = tf.train.Saver()



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(testing_iterator.string_handle())
    sess.run(testing_iterator.initializer)
    test_handle = sess.run(testing_iterator.string_handle())
    saver.restore(sess, os.path.join(model_save_path, model_name))
    print(count_test)

    loss_test_total = 0
    batch_num_test = 0

    #start_time = time.time()
    try:
        batch_num_test = 0
        while True:
            start_time = time.time()
            data_original, data_rec = sess.run([original, prediction], feed_dict = {handle: test_handle, is_training : False})
            end_time = time.time()
            #print((end_time - start_time)/4)
            # print(data_original.shape)
            # print(data_rec.shape)
            batch_num_test = batch_num_test + 1
            if num_output == 1:
                for j in range (data_rec.shape[0]):
                    psnr_cs = functions._psnr(data_original[j, :, :, 0], data_rec[j, :, :, 0])
                    psnr_list_cs.append(psnr_cs)
                    ssim_cs=tf.image.ssim(data_original[j, :, :, 0], data_rec[j, :, :, 0])
                    ssim_list_cs.append(ssim_cs)
            elif num_output == 2:
                for i in range(data_rec.shape[0]):
                    psnr_key_1, psnr_cs = functions._psnr_2d(data_original[i, :, :, :], data_rec[i, :, :, :])
                    psnr_list_key.append(psnr_key_1)
                    psnr_list_cs.append(psnr_cs)
            else:
                if num_output == 4:
                    for i in range(data_rec.shape[0]):
                        psnr_key_1, psnr_cs_1,psnr_cs_2,psnr_cs_3 = functions._psnr_4d(data_original[i, :, :, :], data_rec[i, :, :, :])
                        psnr_list_key.append(psnr_key_1)
                        psnr_list_cs.append(psnr_cs_1)
                        psnr_list_cs.append(psnr_cs_2)
                        psnr_list_cs.append(psnr_cs_3)

                        ssim_key_1, ssim_cs_1, ssim_cs_2, ssim_cs_3 = functions._ssim_4d(data_original[i, :, :, :], data_rec[i, :, :, :])
                        ssim_list_key.append(ssim_key_1)
                        ssim_list_cs.append(ssim_cs_1)
                        ssim_list_cs.append(ssim_cs_2)
                        ssim_list_cs.append(ssim_cs_3)



    except tf.errors.OutOfRangeError:
        pass
    #end_time = time.time()
#print((end_time - start_time)/(count_test*4))
#print(end_time - start_time)
print(batch_num_test)
print(len(psnr_list_key))
print(len(psnr_list_cs))


if num_output == 1:
    a_cs = np.mean(psnr_list_cs)
    print(a_cs)
elif num_output == 2:
    print(len(psnr_list_cs))
    print(len(psnr_list_key) / 3)
    a_cs = np.mean(psnr_list_cs)
    b_key = np.mean(psnr_list_key)
    c_total = (a_cs * len(psnr_list_cs) + b_key * len(psnr_list_key) / 3) / (len(psnr_list_cs) + len(psnr_list_key) / 3)
    print(c_total)
    print(b_key)
    print(a_cs)
else:
    if num_output == 4:
        a_cs = np.mean(psnr_list_cs)
        b_key = np.mean(psnr_list_key)
        c_total = (a_cs * len(psnr_list_cs) + b_key * len(psnr_list_key)) / (len(psnr_list_cs) + len(psnr_list_key))
        print(c_total)
        print(b_key)
        print(a_cs)
        aa_cs = np.mean(ssim_list_cs)
        bb_key = np.mean(ssim_list_key)
        cc_total = (aa_cs * len(ssim_list_cs) + bb_key * len(ssim_list_key)) / (len(ssim_list_cs) + len(ssim_list_key))
        print(cc_total)
        print(bb_key)
        print(aa_cs)

a = data_rec[0,:,:,1]
#a = data_original[0,:,:,1]
plt.imshow(a,cmap ='gray')
plt.show()
img= Image.fromarray(a*255)
img= img.convert('L')
img.save('JsrNet_0_10.jpg')




