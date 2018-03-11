import tensorflow as tf
tf.set_random_seed(777)

x_train = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
y_train = [[1], [2] ,[3]]

X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X: x_train, Y: y_train})
	if step % 200 == 0:
		print(step, cost_val, W_val, b_val)