import tensorflow as tf
import os
import functions_new
from PIL import Image
import numpy as np
import math
import time
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root = '/root/JsrNet 2.0/'

subrate_cs = 0.10
subrate_key = 0.25
batch_size = 32
num_output = 4
block_size = 32
key_size = math.ceil(subrate_key * (block_size ** 2))
cs_size = math.ceil(subrate_cs * (block_size ** 2))

psnr_list_key = []
psnr_list_cs = []
ssim_list_key = []
ssim_list_cs = []
time_list = []
batch_size = 32

data_class = 'test'
filename_test = root + 'TFdata' + '/'  + data_class + '.tfrecord'
model_save_path = root + 'trained_model' + '/' + 'JsrNet'  + '_' + str(subrate_cs) + '.ckpt'



test_set = tf.data.TFRecordDataset(filename_test)
test_set = test_set.map(functions_new.parse_function).batch(batch_size)

sample_test_batch = next(iter(test_set))
print("shape of test_batch_data :", sample_test_batch[0].shape)
print("shape of test_batch_label :",sample_test_batch[1].shape)

model = functions_new.FCNet(key_size, cs_size, block_size)
model.load_weights(model_save_path)

start_time = time.time()

for step, (x,y) in enumerate(test_set):

    pred = model(x)
    # print("shape of pred : ", pred.shape)
    # print("shape of label : ", y.shape)


    for i in range(pred.shape[0]):

        psnr_key_1, psnr_cs_1, psnr_cs_2, psnr_cs_3 = functions_new.psnr_4d(y[i, :, :, :], pred[i, :, :, :])
        psnr_list_key.append(psnr_key_1)
        psnr_list_cs.append(psnr_cs_1)
        psnr_list_cs.append(psnr_cs_2)
        psnr_list_cs.append(psnr_cs_3)

        ssim_key_1, ssim_cs_1, ssim_cs_2, ssim_cs_3 = functions_new.ssim_4d(y[i, :, :, :], pred[i, :, :, :])
        ssim_list_key.append(ssim_key_1)
        ssim_list_cs.append(ssim_cs_1)
        ssim_list_cs.append(ssim_cs_2)
        ssim_list_cs.append(ssim_cs_3)

step = step + 1
print("Number of total batch : ", step)
print("Number of total frames : ", len(psnr_list_key) + len(psnr_list_cs))
print("Number of key frames : ", len(psnr_list_key))
print("Number of cs frames : ", len(psnr_list_cs))



mean_PSNR_cs = np.mean(psnr_list_cs)
mean_PSNR_key = np.mean(psnr_list_key)
mean_PSNR_total = (mean_PSNR_cs * len(psnr_list_cs) + mean_PSNR_key * len(psnr_list_key)) / (len(psnr_list_cs) + len(psnr_list_key))
print("mean PSNR of total frames : ", mean_PSNR_total)
print("mean PSNR of key frames : ", mean_PSNR_key)
print("mean PSNR of cs frames : ", mean_PSNR_cs)

mean_SSIM_cs = np.mean(ssim_list_cs)
mean_SSIM_key = np.mean(ssim_list_key)
mean_SSIM_total = (mean_SSIM_cs * len(ssim_list_cs) + mean_SSIM_key * len(ssim_list_key)) / (len(ssim_list_cs) + len(ssim_list_key))
print("mean SSIM of total frames : ", mean_SSIM_total)
print("mean SSIM of key frames : ", mean_SSIM_key)
print("mean SSIM of cs frames : ", mean_SSIM_cs)

a = pred[0,:,:,1]
plt.imshow(a,cmap ='gray')
plt.show()

# img= Image.fromarray(a*255)
# img= img.convert('L')
# img.save('JsrNet_0_10.jpg')


