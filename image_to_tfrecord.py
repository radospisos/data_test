import tensorflow as tf
import os

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(image, label):
    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecord(image_paths, labels, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for image_path, label in zip(image_paths, labels):
            image = tf.io.read_file(image_path)
            example = serialize_example(image.numpy(), label)
            writer.write(example)

image_dir = '/Users/rodion/Downloads/bird_1_5/test/'
output_file = 'dataset.tfrecord'

image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
labels = [b'c1', b'c2', b'c3', b'c4', b'c5', b'c6', b'c7', b'c8', b'c9', b'c10']

create_tfrecord(image_paths, labels, output_file)