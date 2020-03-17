import tensorflow as tf
from numpy.random import RandomState

#define the size of training data batch
batch_size = 8

#define the argument of nueural network
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

#在維度上使用None,方便使用不同的batch
#the size of dimension should be correct
x = tf.placeholder(tf.float32, shape=(None,2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None,1), name='y-input')

#define forward path of the nueural network
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#define cross entropy and reverse path
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#generating a set of model data randomly
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1,x2) in X]

with tf.Session() as sess:
    #tf.Variable should be initialized before sess.run(...)
    #tf.initialize_all_variables() cannot be used after 2017
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    #set the cycles of training
    STEPS = 10000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
        
        if i % 1000 == 0:    
            print("After %d training step, cross entropy on all data is %g" % (i, total_cross_entropy))
        
        #precisely pointing out how many cycles will reach zero cross entropy
        if total_cross_entropy == 0:
            print("After %d training step, cross entropy on all data is 0" % i)
            break

    print(sess.run(w1))
    print(sess.run(w2))


