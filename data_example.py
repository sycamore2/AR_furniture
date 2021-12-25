import tensorflow as tf
#import functions


def data_example(binary_data,binary_label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'data': bytes_feature(binary_data),
        'label': bytes_feature(binary_label)
    }))
    return example

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))