import tensorflow as tf
tf.set_random_seed(777)

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[2], [2], [2], [1], [1], [1], [0], [0]]

nb_classes = 3


# placeholder 구성
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.int32, shape=[None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])


# 변수 W, b 생성
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# hypothesis, cost function 선언
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

#cross entropy
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

# axis가 0, '열'에 대해 계산
# axis가 1, '행'에 대해 계산
# cost = tf.reduce_mean(tf.reduce_sum(-Y * tf.log(hypothesis), axis=1))
cost = tf.reduce_mean(cost_i)

# learning rate 및 cost minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training model 실행
for step in range(2001):
    _, Y_one, acc = sess.run([optimizer, Y_one_hot, accuracy], feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print(step, Y_one)
        print("Accuracy:", acc)


# Testing & One-hot encoding
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
print(a, sess.run(tf.argmax(a, 1)))

