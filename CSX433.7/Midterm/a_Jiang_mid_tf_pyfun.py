#####Information section#########
## Name:  Jiang Li
## Email: riverlee2008@gmail.com
## Class: Tensorflow
###############################
import numpy as np
import tensorflow as tf

## User defined input
n = int(raw_input("Enter the length of your random vector: "))
w = int(raw_input("Enter the length of window width:"))


## Randomly initialize original vector
a = np.random.normal(loc=1,scale=2,size=n)


## function to do SMA operation
def sma(a,width=2):
    ## Get last index
    last_index = len(a)-width+1
    return np.array([np.mean(a[i:(i+width)]) for i in range(last_index)])


## Use Tensorflow
with tf.name_scope("Input_placeholder"):
    tf_a = tf.placeholder(shape=[n],dtype=tf.float32)

with tf.name_scope("SMA_Vector"):
    tf_w = tf.constant(w,dtype=tf.int32)
    tf_sma = tf.py_func(sma,[tf_a,tf_w],tf.float32)

with tf.name_scope("SMA_summary"):
    tf_sma_mean = tf.reduce_mean(tf_sma)

## write out graph
writer =tf.summary.FileWriter("./Midterm_graph_pyfun",graph = tf.get_default_graph())
writer.close()

with tf.Session() as sess:
    outa =  sess.run(tf_a,feed_dict={tf_a:a})
    outsma = sess.run(tf_sma,feed_dict={tf_sma: sma(a)})
    outsma_mean = sess.run(tf_sma_mean,feed_dict={tf_sma: sma(a)})
    print("Our vector is:",outa)
    print("SMA vector is:",outsma)
    print("Total Mean of SMA vector is",outsma_mean)



         
