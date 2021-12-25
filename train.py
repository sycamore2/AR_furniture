import tensorflow as tf
import os
import functions
import time
import tensorflow.contrib.slim as slim
import math


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root = '/root/DINet/'

subrate_key = 0.25
subrate_cs = 0.04
pre_mode = 1
num_output = 4

pre_model_save_path = root + 'pre_trained_model' + '/' + str(subrate_key)
pre_model_name = 'model_rec3_25_2_voc300.ckpt'
#################################################################################################
if num_output == 2:
    model_save_path = root + 'trained_model' + '/' + 'trained_model_2_output' + '/' + str(subrate_cs)
    filename_train = root + 'TFdata' + '/' + 'TFdata_test_1' + '/' + 'TFdata_2_output' + '/' + 'train' + '.tfrecord'
    filename_validation = root + 'TFdata' + '/' + 'TFdata_test_1' + '/' + 'TFdata_2_output' + '/' + 'validation' + '.tfrecord'

else:
    if num_output == 4:
        model_save_path = root + 'trained_model' + '/' + 'trained_model_4_output' + '/' + str(subrate_cs)
        filename_train = root + 'TFdata' + '/' + 'TFdata_test_1' + '/' + 'TFdata_4_output' + '/' + 'train' + '.tfrecord'
        filename_validation = root + 'TFdata' + '/' + 'TFdata_test_1' + '/' + 'TFdata_4_output' + '/' + 'validation' + '.tfrecord'




# model_save_path = root + 'trained_model' + '/' + 'trained_model_4_output' + '/' + str(subrate_cs)
# filename_train = root + 'TFdata' + '/' + 'TFdata_test_1' + '/' + 'TFdata_2_output' + '/'+ 'train' + '.tfrecord'
# filename_validation = root + 'TFdata' + '/' + 'TFdata_test_1' + '/' +  'TFdata_2_output' + '/'+ 'validation' + '.tfrecord'
#######################################################################################33
#filename_train = root + 'TFdata' + '/' +'TFdata_key_rec' + '/' + 'TFdata_1_output' + '/'+ 'train' + '.tfrecord'
#filename_validation = root + 'TFdata' + '/' + 'TFdata_key_rec' + '/' + 'TFdata_1_output' + '/'+ 'validation' + '.tfrecord'
batch_size = 16
batch_size_val = 16
block_size = 32
weight_decay_rate = 1e-4
learning_rate_base = 1e-4
Learning_rate_decay_step = 500
learning_rate_decay_rate = 1
epoch_num = 50
print_interval = 2
#model_save_path = root + 'trained_model' + '/' + 'trained_model_3_output' + '/' + str(subrate_cs)
# model_save_path = root + 'trained_model' + '/' + 'trained_model_1_output' + '/' + str(subrate_cs)
model_name = 'DINet'  + '_' + str(subrate_cs)


is_training = tf.placeholder(tf.bool)
handle = tf.placeholder(tf.string, shape=[])
count_train = functions.count_dataset(filename_train)
count_validation = functions.count_dataset(filename_validation)
train_set = tf.data.TFRecordDataset(filename_train)
validation_set = tf.data.TFRecordDataset(filename_validation)
train_set = train_set.map(functions.parse_function)
validation_set = validation_set.map(functions.parse_function)
train_set = train_set.shuffle(count_train, reshuffle_each_iteration=True)
# train_set = train_set.batch(batch_size,drop_remainder=True)
train_set = train_set.batch(batch_size,drop_remainder=False)
validation_set = validation_set.shuffle(count_validation, reshuffle_each_iteration=True)
# validation_set = validation_set.batch(batch_size_val,drop_remainder=True)
validation_set = validation_set.batch(batch_size_val,drop_remainder=False)
iterator = tf.data.Iterator.from_string_handle(handle, train_set.output_types, train_set.output_shapes)
data, label = iterator.get_next()
training_iterator = train_set.make_initializable_iterator()
validation_iterator = validation_set.make_initializable_iterator()

global_step = tf.Variable(0,
                          dtype=tf.int32,
                          trainable=False,
                          name='global_step')
#
#
prediction = functions.DINet_4(data, block_size, subrate_key, subrate_cs, weight_decay_rate, is_training)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     train_handle = sess.run(training_iterator.string_handle())
#     validation_handle = sess.run(validation_iterator.string_handle())
#
#     sess.run(training_iterator.initializer)
#     print(count_train)
#     print(count_validation)
#     ddd = sess.run(prediction, feed_dict={handle: train_handle})
#     # aaa=sess.run(tf.reshape(ddd[0],[160,160]))
#
# print(ddd.shape)

reg_loss = tf.losses.get_regularization_loss()

#
#
#loss_without_regularization = tf.losses.mean_squared_error(prediction, label)
loss_without_regularization = functions.myloss(prediction, label ,reg_loss)

learning_rate = tf.train.exponential_decay(learning_rate_base,
                                           global_step,
                                           Learning_rate_decay_step,
                                           learning_rate_decay_rate,
                                           staircase=True)
#decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_without_regularization, global_step=global_step)

saver = tf.train.Saver()


if pre_mode == 1:
    # restore pretrain
    var = tf.global_variables()
    # var_to_restore = [val for val in var if 'conv0' in val.name or 'conv999' in val.name]
    var_to_restore = [val for val in var if 'conv' in val.name]
    saver_pre = tf.train.Saver(var_to_restore)

# elif pre_mode == 2:
#     # continue ckpt
#     ckpt = tf.train.get_checkpoint_state(settings.model_save_path)
#     print('d' + ckpt.model_checkpoint_path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    if pre_mode == 1:
        # restore pre
        # saver = tf.train.import_meta_graph('/root/DINet/pre_trained_model/0.01/model_rec3_01_2_voc300.ckpt.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('/root/DINet/pre_trained_model/0.01'))

        saver_pre.restore(sess, os.path.join(pre_model_save_path, pre_model_name))



    train_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())

    print(count_train)
    print(count_validation)


    batch_num_train_2 = 0
    loss_train_total_2 = 0
    start_time_2 = time.time()

    for i in range(epoch_num):

        ### train
        sess.run(training_iterator.initializer)
        try:
            loss_train_total_1 = 0
            batch_num_train_1 = 0
            start_time = time.time()
            while True:
                _, loss_train, step, lr = \
                    sess.run([optimizer, loss_without_regularization, global_step, learning_rate],
                             feed_dict={handle: train_handle, is_training: True})
                loss_train_total_1 += loss_train
                batch_num_train_1 += 1

                if batch_num_train_2 % print_interval == 0:
                    loss_train_total_2 = 0
                    loss_wor_total_2 = 0
                    batch_num_train_2 = 0
                    start_time_2 = time.time()
        except tf.errors.OutOfRangeError:
            pass

        print('tra epoch: {0}, step: {1}, time: {2:.2f}, lr: {3:.2e},  loss: {4:.4f}'
              .format(i,
                      step,
                      round(time.time() - start_time),
                      lr,
                      10000 * loss_train_total_1 / batch_num_train_1))

        #validation
        sess.run(validation_iterator.initializer)
        start_time = time.time()
        try:
            loss_val_total_1 = 0
            batch_num_val_1 = 0

            while True:
                loss_val, step = sess.run([  loss_without_regularization, global_step],
                                                        feed_dict={handle: validation_handle, is_training: False})
                loss_val_total_1 += loss_val
                batch_num_val_1 += 1

        except tf.errors.OutOfRangeError:
            pass

        print(
            'val epoch: {0}, step: {1}, time: {2:.2f},val loss: {3:.4f}'
                .format(i, step, round(time.time() - start_time),
                         10000*loss_val_total_1 / batch_num_val_1  ))

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
        saver.save(sess, os.path.join(model_save_path, model_name))
        save_switch = 0
        print('----------------' + ': save' + '----------------')
    else:
        saver.save(sess, os.path.join(model_save_path, model_name))
        print('----------------'  + ': save' + '----------------')
        # print('----------------' + str(i) + ': save' + '----------------')
        # if not os.path.exists(model_save_path):
        #     os.makedirs(model_save_path)
        # saver.save(sess, os.path.join(model_save_path, model_name))
        # save_switch = 0
        # print('----------------' + str(i) + ': save' + '----------------')




