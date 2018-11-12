import tensorflow as tf
import pandas as pd
import numpy as np
from time import time
import test

n_input = 12
train_person_id = input("Enter person's id : ")
# test_image_path = input("Enter path of signature image : ")

train_path = 'data/Features/Training/training_'+train_person_id+'.csv'
# test.testing(test_image_path)
# test_path = 'data/TestFeatures/testcsv.csv'
test_path = 'data/Features/Testing/testing_'+train_person_id+'.csv'

def readCSV(train_path, test_path, type2=False):
    # Reading train data
    df = pd.read_csv(train_path, usecols=range(n_input))
    train_input = np.array(df.values)
    train_input = train_input.astype(np.float32, copy=False)  # Converting input to float_32
    
    df = pd.read_csv(train_path, usecols=(n_input,))
    temp = [elem[0] for elem in df.values]
    correct = np.array(temp)
    corr_train = np.eye(2)[correct]      # Converting to one hot
    # Reading test data
    df = pd.read_csv(test_path, usecols=range(n_input))
    test_input = np.array(df.values)
    test_input = test_input.astype(np.float32, copy=False)
    if not(type2):
        df = pd.read_csv(test_path, usecols=(n_input,))
        temp = [elem[0] for elem in df.values]
        correct = np.array(temp)
        corr_test = np.eye(2)[correct]      # Converting to one hot
    if not(type2):
        return train_input, corr_train, test_input, corr_test
    else:
        return train_input, corr_train, test_input

tf.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 1700
display_step = 1

# Network Parameters
n_hidden_1 = 7 # 1st layer number of neurons
# n_hidden_2 = 10 # 2nd layer number of neurons
# n_hidden_3 = 30 # 3rd layer
n_classes = 2 # no. of classes (genuine or forged)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], seed=2))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
#     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#     'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes], seed=4))
}


# Create model
def multilayer_perceptron(x):
    layer_1 = tf.tanh((tf.matmul(x, weights['h1']) + biases['b1']))
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer

loss_op = tf.reduce_mean(tf.squared_difference(logits, Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# For accuracies
pred = tf.nn.softmax(logits)  # Apply softmax to logits
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()

def evaluate(train_path, test_path, type2=False):   
    if not(type2):
        train_input, corr_train, test_input, corr_test = readCSV(train_path, test_path)
    else:
        train_input, corr_train, test_input = readCSV(train_path, test_path, type2)
    ans = 'Random'
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
            if cost<0.0001:
                break
#             # Display logs per epoch step
#             if epoch % 999 == 0:
#                 print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(cost))
#         print("Optimization Finished!")
        
        # Finding accuracies
        accuracy1 =  accuracy.eval({X: train_input, Y: corr_train})
#         print("Accuracy for train:", accuracy1)
#         print("Accuracy for test:", accuracy2)
        if type2 is False:
            accuracy2 =  accuracy.eval({X: test_input, Y: corr_test})
            return accuracy1, accuracy2
        else:
            prediction = pred.eval({X: test_input})
            if prediction[0][1]>prediction[0][0]:
                print('Genuine Image')
                return True
            else:
                print('Forged Image')
                return False


def trainAndTest(rate=0.001, epochs=1700, neurons=7, display=False):    
    print('aya pochya')
    start = time()
    
    # Parameters
    global training_rate, training_epochs, n_hidden_1
    learning_rate = rate
    training_epochs = epochs

    # Network Parameters
    n_hidden_1 = neurons # 1st layer number of neurons
    # n_hidden_2 = 7 # 2nd layer number of neurons
    # n_hidden_3 = 30 # 3rd layer

    train_avg, test_avg = 0, 0
    n = 11
    # flag=[True] * 5000
    for i in range(1,n+1):
        # flag.append(True);
        if display:
            print("Running for Person id",i)
        temp = ('0'+str(i))[-2:]
        curr_tr_pth=train_path.replace('01',temp)
        curr_ts_pth=test_path.replace('01',temp)
        print(curr_tr_pth)
        print(curr_ts_pth)
        train_score, test_score = evaluate(curr_tr_pth,curr_ts_pth)
        train_avg += train_score
        test_avg += test_score
    if display:
#         print("Number of neurons in Hidden layer-", n_hidden_1)
        print("Training average-", train_avg/n)
        print("Testing average-", test_avg/n)
        print("Time taken-", time()-start)
    return train_avg/n, test_avg/n, (time()-start)/n


trainAndTest(display=True)
#print(test_path)
# evaluate(train_path, test_path,type2=True)
print('pati gayu')
