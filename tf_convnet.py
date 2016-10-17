"""
Start pure TensorFlow version.
"""
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split

with open('train.p', mode='rb') as f:
    train = pickle.load(f)
with open('test.p', mode='rb') as f:
    test = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(train['features'], train['labels'], test_size=0.33, random_state=0)
X_test, y_test = test['features'], test['labels']

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

batch_size = 64

# 0-255 -> 0-1
X_train /= 255
X_val /= 255
X_test /= 255

# number of traffic signs
n_classes = 43
epochs = 10
input_shape = X_train.shape[1:]

feature_shape = (32, 32, 3)
feature_size = 32*32*3

# CONV1
features = tf.placeholder(tf.float32, (batch_size,) + feature_shape, name="features")
filters = tf.Variable(tf.truncated_normal([3,3,3,32]))
conv1 = tf.nn.conv2d(features, filters, strides=(1,2,2,1), padding="VALID")
relu = tf.nn.relu(conv1)
relu_shape = relu.get_shape().as_list()
reshape = tf.reshape(relu, [relu_shape[0], relu_shape[1]*relu_shape[2]*relu_shape[3]])

# FC2
feature_size = feature_shape[0] // 3 * feature_shape[1] // 3 * 32
W1 = tf.Variable(tf.random_normal((feature_size, 100), mean=0, stddev=0.01) , name="W1")
b1 = tf.Variable(tf.zeros((100)) , name="b1")

# FC3
W2 = tf.Variable(tf.random_normal((100, n_classes), mean=0, stddev=0.01) , name="W2")
b2 = tf.Variable(tf.zeros((n_classes)) , name="b2")

labels = tf.placeholder(tf.int64, (None), name="labels")

out1 = tf.matmul(reshape, W1) + b1
out1 = tf.nn.relu(out1)
logits = tf.matmul(out1, W2) + b2

correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels), 0)
optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(epochs):
		for offset in range(0, X_train.shape[0], batch_size):
			end = offset + batch_size
			batch_X = X_train[offset:end]
			batch_y = y_train[offset:end]	

			if batch_X.shape[0] != batch_size:
				continue

			l, _ = sess.run([loss, train_op], feed_dict={features: batch_X, labels: batch_y})

		val_l, val_acc = sess.run([loss, accuracy], feed_dict={features: X_val, labels: y_val})
		print("Validation	Loss =", val_l)
		print("Validation	Accuracy =", val_acc)

	
	#test_l, test_acc = sess.run([loss, accuracy], feed_dict={features: X_test, labels: y_test})
	#print("Testing	Loss =", test_l)
	#print("Testing	Accuracy =", test_acc)


