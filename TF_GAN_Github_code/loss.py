import tensorflow as tf
import os
import random
import numpy as np

def  l1_loss_reconstruction(real, reconstructed):

    return tf.reduce_mean(tf.abs(real - reconstructed))

def l1_loss_identity(real, identity):

    return tf.reduce_mean(tf.abs(real -  identity))



def combined_loss(target, sig, rel , sel , elu ,lrel):
    
   
    var_1=((1.10*tf.reduce_mean(tf.square(target - sig))) + (1.5*tf.reduce_mean(tf.abs(target - sig))))

    var_2=((1.01*tf.reduce_mean(tf.square(target - rel))) + (1.5*tf.reduce_mean(tf.abs(target - rel))))
    var_3=((1.01*tf.reduce_mean(tf.square(target - sel))) + (1.5*tf.reduce_mean(tf.abs(target - sel))))
    var_4=((1.01*tf.reduce_mean(tf.square(target - elu))) + (1.5*tf.reduce_mean(tf.abs(target - elu))))
    var_5=((1.01*tf.reduce_mean(tf.square(target - lrel))) + (1.5*tf.reduce_mean(tf.abs(target - lrel))))
    
    var_i=tf.math.minimum(var_1,var_2)
    var_j=tf.math.minimum(var_i,var_3)
    var_k=tf.math.minimum(var_j,var_4)
    minimum=tf.math.minimum(var_k,var_5)
    
    return minimum



