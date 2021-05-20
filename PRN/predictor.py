import tensorflow as t
import tensorflow.contrib.layers as cl
from tensorflow.contrib.framework import arg_scope
import numpy as np

def rb(x, num_out, kernel_size = 4, stride=1, a_fn=t.nn.relu, n_fn=cl.batch_norm, scope=None):
    assert num_out%2==0
    with t.variable_scope(scope, 'resBlock'):
        shortcut = x
        if stride != 1 or x.get_shape()[3] != num_out:
            shortcut = cl.conv2d(shortcut, num_out, kernel_size=1, stride=stride, 
                        activation_fn=None, normalizer_fn=None, scope='shortcut')
        x = cl.conv2d(x, num_out/2, kernel_size=1, stride=1, padding='SAME')
        x = cl.conv2d(x, num_out/2, kernel_size=kernel_size, stride=stride, padding='SAME')
        x = cl.conv2d(x, num_out, kernel_size=1, stride=1, activation_fn=None, padding='SAME', normalizer_fn=None)

        x += shortcut       
        x = n_fn(x)
        x = a_fn(x)
    return x


class resfcn256(object):
    def __init__(self, resolution_input = 256, resolution_output = 256, channel = 3, name = 'resfcn256'):
        self.name = name
        self.channel = channel
        self.resolution_input = resolution_input
        self.resolution_output = resolution_output

    def __call__(self, x, is_training = True):
        with t.variable_scope(self.name) as scope:
            with arg_scope([cl.batch_norm], is_training=is_training, scale=True):
                with arg_scope([cl.conv2d, cl.conv2d_transpose], activation_fn=t.nn.relu, 
                                     normalizer_fn=cl.batch_norm, 
                                     biases_initializer=None, 
                                     padding='SAME',
                                     weights_regularizer=cl.l2_regularizer(0.0002)):
                    size = 16  
                    
                    se = cl.conv2d(x, num_outputs=size, kernel_size=4, stride=1) 
                    se = rb(se, num_out=size * 2, kernel_size=4, stride=2) 
                    se = rb(se, num_out=size * 2, kernel_size=4, stride=1)
                    se = rb(se, num_out=size * 4, kernel_size=4, stride=2) 
                    se = rb(se, num_out=size * 4, kernel_size=4, stride=1) 
                    se = rb(se, num_out=size * 8, kernel_size=4, stride=2) 
                    se = rb(se, num_out=size * 8, kernel_size=4, stride=1) 
                    se = rb(se, num_out=size * 16, kernel_size=4, stride=2)
                    se = rb(se, num_out=size * 16, kernel_size=4, stride=1) 
                    se = rb(se, num_out=size * 32, kernel_size=4, stride=2)
                    se = rb(se, num_out=size * 32, kernel_size=4, stride=1) 

                    pd = cl.conv2d_transpose(se, size * 32, 4, stride=1) 
                    pd = cl.conv2d_transpose(pd, size * 16, 4, stride=2)  
                    pd = cl.conv2d_transpose(pd, size * 16, 4, stride=1)  
                    pd = cl.conv2d_transpose(pd, size * 16, 4, stride=1)  
                    pd = cl.conv2d_transpose(pd, size * 8, 4, stride=2)  
                    pd = cl.conv2d_transpose(pd, size * 8, 4, stride=1)  
                    pd = cl.conv2d_transpose(pd, size * 8, 4, stride=1)  
                    pd = cl.conv2d_transpose(pd, size * 4, 4, stride=2)  
                    pd = cl.conv2d_transpose(pd, size * 4, 4, stride=1)
                    pd = cl.conv2d_transpose(pd, size * 4, 4, stride=1) 
                    
                    pd = cl.conv2d_transpose(pd, size * 2, 4, stride=2)
                    pd = cl.conv2d_transpose(pd, size * 2, 4, stride=1)
                    pd = cl.conv2d_transpose(pd, size, 4, stride=2) 
                    pd = cl.conv2d_transpose(pd, size, 4, stride=1) 

                    pd = cl.conv2d_transpose(pd, 3, 4, stride=1) 
                    pd = cl.conv2d_transpose(pd, 3, 4, stride=1)
                    pos = cl.conv2d_transpose(pd, 3, 4, stride=1, activation_fn = t.nn.sigmoid)
                                
                    return pos
    @property
    def vars(self):
        return [var for var in t.global_variables() if self.name in var.name]


class FacePointPositionPredictor():
    def __init__(self, resolution_input = 256, resolution_output = 256): 

        self.resolution_input = resolution_input
        self.resolution_output = resolution_output
        self.MaxPos = resolution_input*1.1

        self.network = resfcn256(self.resolution_input, self.resolution_output)


        self.x = t.placeholder(t.float32, shape=[None, self.resolution_input, self.resolution_input, 3])  
        self.x_op = self.network(self.x, is_training = False)
        self.sess = t.Session(config=t.ConfigProto(gpu_options=t.GPUOptions(allow_growth=True)))

    def restore(self, model_path):        
        t.train.Saver(self.network.vars).restore(self.sess, model_path)
 
    def predict(self, image):
        pos = self.sess.run(self.x_op, 
                    feed_dict = {self.x: image[np.newaxis, :,:,:]})
        pos = np.squeeze(pos)
        return pos*self.MaxPos

    def predict_batch(self, images):
        pos = self.sess.run(self.x_op, 
                    feed_dict = {self.x: images})
        return pos*self.MaxPos

