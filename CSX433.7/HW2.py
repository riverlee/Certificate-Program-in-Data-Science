#!/usr/bin/env python
######################################
# User: Jiang Li
# Date: Sat Jun  9 17:00:00 PDT 2018
######################################

"""
HW2:
Recreate the graph and visualize it in Tensorboard using:
1.  Placeholder for an input array with dtype float32 and shape None
2.  Scopes for the input, middle section and final node f
3.  Feed the placeholder with an array A consis=ng of 100 normally distributed random numbers with Mean = 1 and Standard deviation = 2
4.  Save your graph and show it in TensorBoard
5.  Plot you input array on a separate figure
6.  Make sure you comment your code well and provide your name on top of your work
7.  Email your Github link (or code directly) to me including your .py file + screenshots of TensorBoard

"""

## 1. Load package for the assignment
import numpy as np
import tensorflow as tf

#### Input layer
with tf.name_scope("Input_placeholder"):
    a = tf.placeholder(tf.float32,shape=None,name = "input")

#### Midle layer
with tf.name_scope("Middle_session"):
    b = tf.reduce_prod(a,name = "prob")
    c = tf.reduce_mean(a,name = "mean")
    d = tf.reduce_sum(a,name = "sum")
    e = tf.add(b,c,name = "add")
   
#### Final layer
with tf.name_scope("Final_node"):
    f = tf.multiply(d,e,name = "mul")
    

writer =tf.summary.FileWriter("./HW2_graph",graph = tf.get_default_graph())
writer.close()
