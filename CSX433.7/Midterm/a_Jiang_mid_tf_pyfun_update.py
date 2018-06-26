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
graph = tf.Graph()

with graph.as_default():
    with tf.name_scope("variables"):
        global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name="global_step")
        
        total_output=tf.Variable(0.0,dtype=tf.float32,trainable=False,name="total_output"
    
    with tf.name_scope("transformation"):
        with tf.name_scope("Input_placeholder"):
            tf_a = tf.placeholder(shape=[None],dtype=tf.float32,name="input")

        with tf.name_scope("SMA_Vector"):
            tf_w = tf.constant(w,dtype=tf.int32)
            tf_sma = tf.py_func(sma,[tf_a,tf_w],tf.float32)

        with tf.name_scope("SMA_summary"):
            tf_sma_mean = tf.reduce_mean(tf_sma)
            
    with tf.name_scope("update"):
        update_total = total_output.assign_add(output)
        increasement_step = global_step.assign_add(1)
        
    with tf.name_scope("Summaries"):
        tf.summary.scalar("Output",tf_sma_mean)
        
    with tf.name_scope("global_ops"):
        init = tf.initialize_all_variables()
        merged_summaries = tf.summary.merge_all()
        

## write out graph

with tf.Session(graph=graph) as sess:
    writer =tf.summary.FileWriter("./Midterm_graph_pyfun",graph =graph)
    writer.close()
    
    sess.run(init)
    
    outa =  sess.run(tf_a,feed_dict={tf_a:a})
    outsma = sess.run(tf_sma,feed_dict={tf_sma: sma(a)})
    outsma_mean = sess.run(tf_sma_mean,feed_dict={tf_sma: sma(a)})
    print("Our vector is:",outa)
    print("SMA vector is:",outsma)
    print("Total Mean of SMA vector is",outsma_mean)



         
