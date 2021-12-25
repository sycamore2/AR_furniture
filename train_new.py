import tensorflow as tf
import os
import time
import math
from    tensorflow import keras
from    tensorflow.keras import  layers
import functions_new




os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root = '/root/JsrNet 2.0/'


subrate_key = 0.25##关键帧采样率
subrate_cs = 0.04##非关键帧采样率
num_output = 4
batch_size = 32
batch_size_val = 16
block_size = 32
learning_rate = 1e-4
epoch_num = 2
model_name = 'JsrNet'  + '_' + str(subrate_cs) + '.ckpt'
print(model_name)

key_size = math.ceil(subrate_key * (block_size ** 2))
cs_size = math.ceil(subrate_cs * (block_size ** 2))



if num_output == 2:
    model_save_path = root + 'trained_model' + '/' + 'JsrNet'  + '_' + str(subrate_cs) + '.ckpt'
    filename_train = root + 'TFdata' + '/' + 'train' + '.tfrecord'
    filename_validation = root + 'TFdata' + '/' + 'validation' + '.tfrecord'

else:
    if num_output == 4:
        model_save_path = root + 'trained_model' + '/' + 'JsrNet'  + '_' + str(subrate_cs) + '.ckpt'
        filename_train = root + 'TFdata' + '/' + 'train' + '.tfrecord'
        filename_validation = root + 'TFdata' + '/' + 'validation' + '.tfrecord'




train_set = tf.data.TFRecordDataset(filename_train)
train_set = train_set.shuffle(100000).map(functions_new.parse_function).batch(batch_size)

validation_set = tf.data.TFRecordDataset(filename_validation)
validation_set = validation_set.shuffle(100000).map(functions_new.parse_function).batch(batch_size)



sample_train_batch = next(iter(train_set))
sample_validation_batch = next(iter(validation_set))
print("shape of train_batch_data :", sample_train_batch[0].shape)
print("shape of train_batch_label :",sample_train_batch[1].shape)
print("shape of validation_batch_data :", sample_validation_batch[0].shape)
print("shape of validation_batch_label :",sample_validation_batch[1].shape)




model = functions_new.FCNet(key_size, cs_size, block_size)
model.build(input_shape=(None, 160, 160, 4))


optimizer = tf.optimizers.Adam(lr=learning_rate)



for epoch in range(epoch_num):


    # train
    start_time = time.time()

    for step_train, (x_train,y_train) in enumerate(train_set):

        with tf.GradientTape() as tape:
            rec_train = model(x_train)
            loss_mse_train = tf.reduce_mean(tf.losses.MSE(y_train, rec_train))


        grads = tape.gradient(loss_mse_train, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    time_train = round(time.time() - start_time)


    # validation

    start_time = time.time()
    loss_total_validation = 0

    for  step_validation, (x_validation, y_validation) in enumerate(validation_set):

        rec_validation = model(x_validation)
        loss_mse_validation = tf.reduce_mean(tf.losses.MSE(y_validation, rec_validation))
        loss_total_validation = loss_total_validation + loss_mse_validation

    time_validation = round(time.time() - start_time)

    print("epoch : ", epoch, ", time_train : ", time_train, ", mse_train : ", float(loss_mse_train))
    print("epoch : ", epoch, ", time_validation : ", time_validation, ", mse_validation : ", float(loss_total_validation/(step_validation + 1)))

        # if step_train % 10 == 0:
        #     print("epoch : ", epoch, ", step : ", step_train, ", time : ", round(time.time() - start_time), "mse : ", float(loss_mse_train))




#
model.save_weights(model_save_path)
print("saved total model.")