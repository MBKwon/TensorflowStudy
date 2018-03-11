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
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# placeholder 구성
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 3])

# 변수 W, b 생성
W = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_normal([3]), name='bias')

# hypothesis, cost function 선언
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# axis가 0, '열'에 대해 계산
# axis가 1, '행'에 대해 계산
cost = tf.reduce_mean(tf.reduce_sum(-Y * tf.log(hypothesis), axis=1))

# learning rate 및 cost minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training model 실행
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, optimizer], feed_dict={X:x_data, Y:y_data})
    if step % 200 == 0:
        print(step, cost_val, W_val, b_val)


# Testing & One-hot encoding
a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
print(a, sess.run(tf.argmax(a, 1)))

