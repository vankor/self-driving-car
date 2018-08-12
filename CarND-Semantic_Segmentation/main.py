import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    default_graph = tf.get_default_default_graph()
    image_input = default_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Vgg 7 layer 1x1  convolution
    convolutional_1 = tf.layers.conv2d(vgg_layer7_out,
                              num_classes, 1,
                              padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Upsample
    output_1 = tf.layers.conv2d_transpose(convolutional_1, num_classes, 4, 2,
                                         padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Vgg 4 layer 1x1 convolution
    convolutional_2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    # Skip layer
    skip_out_1 = tf.add(convolutional_2, output_1)

    # Upsample
    output_2 = tf.layers.conv2d_transpose(skip_out_1, num_classes, 4, 2, padding='same',
                                         kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


    # Vgg 3 layer 1x1 convolution
    convolutional_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                              kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))


    # Skip layer
    skip_out_2 = tf.add(convolutional_3, output_2)
    
    # Upsample
    output = tf.layers.conv2d_transpose(skip_out_2, num_classes, 16, 8, padding='same',
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    correct_label = tf.reshape(correct_label, (-1, num_classes))

    logits = tf.reshape(nn_last_layer, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits,
                                                                                labels= correct_label))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             real_output, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param real_output: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print("Training process...")
    print()
    global training_error
    for i in range(epochs):
        for image, label in get_batches_fn(batch_size):
            _, training_error = sess.run([train_op, cross_entropy_loss], feed_dict={input_image:image,
                                                                         real_output:label,
                                                                         keep_prob:0.6,
                                                                         learning_rate: 0.0001})

        print("EPOCH {} ...".format(i+1))
        print("Training error = {:.3f}".format(training_error))
        print()

    print("Network is trained...")
    print()

tests.test_train_nn(train_nn)

def run():

    data_dir = './data'
    runs_dir = './runs'

    num_classes = 2

    image_shape = (160, 576)

    tests.test_for_kitti_dataset(data_dir)

    # Pre trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        
        epochs = 50
        
        batch_size = 7

        # Placeholders
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        real_output = tf.placeholder(tf.int32, [None, None, None, num_classes], name='real_output')

        # Get batches function
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # VGG path
        vgg_path = os.path.join(data_dir, 'vgg')

        # Construct neural network using loaded vgg, custom layers and optimizer
        input_image, keep_prob, layer3_out_vgg, layer4_out_vgg, layer7_out_vgg = load_vgg(sess, vgg_path)
        layer_output = layers(layer3_out_vgg, layer4_out_vgg, layer7_out_vgg, num_classes)

        logits, train_op, cross_entropy_loss = optimize(layer_output, real_output, learning_rate, num_classes)

        # Train network
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
                 input_image, real_output, keep_prob, learning_rate)

        # Saving inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

if __name__ == '__main__':
    run()
