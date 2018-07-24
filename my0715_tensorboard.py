import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("..", one_hot=True)
print(mnist.train.images.shape)

learning_rate = 0.001
training_epochs = 1
training_epochs = 15
batch_size = 100

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10
n_samples = mnist.train.num_examples


with tf.name_scope('inputs'):
    x = tf.placeholder("float", [None, n_input], name='input_x')
    y = tf.placeholder("float", [None, n_classes], name='input_y')

with tf.name_scope('input_reshape'):
    image_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_input, 10)


def add_layer(x, input_tensors, output_tensors, layer_name, activation_function=None):
    with tf.name_scope('Layer'):
        with tf.name_scope('Weights'):
            weight = tf.Variable(tf.random_normal([input_tensors, output_tensors]), name='w')
            tf.summary.histogram(name=layer_name + '/Weights', values=weight)
        with tf.name_scope('Bias'):
            bias = tf.Variable(tf.random_normal([output_tensors]), name='b')
            tf.summary.histogram(name=layer_name + '/Bias', values=bias)
        with tf.name_scope('Wx_plus_b'):
            formula = tf.add(tf.matmul(x, weight), bias)
        if activation_function is None:
            outputs = formula
        else:
            outputs = activation_function(formula)
        tf.summary.histogram(name=layer_name + '/Outputs', values=outputs)
        return outputs


layer1 = add_layer(x, input_tensors=n_input, output_tensors=n_hidden_1, layer_name='layer1',
                   activation_function=tf.nn.relu)
layer2 = add_layer(layer1, input_tensors=n_hidden_1, output_tensors=n_hidden_2, layer_name='layer2',
                   activation_function=tf.nn.relu)
out_layer = add_layer(layer2, input_tensors=n_hidden_2, output_tensors=n_classes, layer_name='out_layer',
                      activation_function=None)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_layer, labels=y))
    tf.summary.scalar('loss', cost)

with tf.name_scope('optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(out_layer, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    tf.summary.scalar("accuracy", acc)

init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)

    ## Merge Summary

    writer = tf.summary.FileWriter("tensorboard/", graph=sess.graph)

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c, result = sess.run([optimizer, cost, merged], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
            ## Adding summary of each step
            writer.add_summary(result, epoch * total_batch + i)

        print("Epoch: {} cost={}".format(epoch + 1, avg_cost))

    print("Training Completed in {} Epochs".format(training_epochs))
