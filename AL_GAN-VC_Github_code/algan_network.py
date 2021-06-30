import tensorflow as tf 

def conv1d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides = 1, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv1d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def conv2d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer


def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)
    
    return activation

def upsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'upsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def downsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    shuffle_size = 1,
    name_prefix = 'downsample1d_block_'):
    
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def upsample2d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'upsample2d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def pixel_shuffler(inputs, shuffle_size = 1, name = None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs

def instance_norm_layer(
    inputs, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer


def residual1d_block(
    inputs, 
    filters = 1024, 
    kernel_size = 3, 
    strides = 1,
    name_prefix = 'residule_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    h2 = conv1d_layer(inputs = h1_glu, filters = filters , kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')
    h2_gates = conv1d_layer(inputs = h1_glu, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_gates')
    h2_norm_gates = instance_norm_layer(inputs = h2_gates, activation_fn = None, name = name_prefix + 'h2_norm_gates')
    h2_glu = gated_linear_layer(inputs = h2_norm, gates = h2_norm_gates, name = name_prefix + 'h2_glu')

    h3 = inputs + h1_glu + h2_glu
   
    return h3



def generator_alpha2beta(inputs, reuse = False, scope_name = 'generator_alpha2beta'):
   
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')
   
    with tf.variable_scope(scope_name) as scope:
       
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h0 = conv1d_layer(inputs = inputs, filters = 24, kernel_size = 5, strides = 1, activation = None , name = 'h0_conv')
        h0_gates = conv1d_layer(inputs = inputs, filters = 24, kernel_size = 5, strides = 1, activation = None, name = 'h0_conv_gates')
        h0_glu = gated_linear_layer(inputs = h0, gates = h0_gates, name = 'h0_glu')
       
        h1 = conv1d_layer(inputs = h0_glu, filters = 64, kernel_size = 5, strides = 1, activation = None , name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = h0_glu, filters = 64, kernel_size = 5, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')
        

        # upsample
        u1 = upsample1d_block(inputs = h1_glu, filters = 64, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block1_')
        u2 = upsample1d_block(inputs = u1, filters = 128, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block2_')
        u3 = upsample1d_block(inputs = u2, filters = 256, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block3_')
        u4 = upsample1d_block(inputs = u3, filters = 512, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block4_')
        u5 = upsample1d_block(inputs = u4, filters = 1024, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block5_')
        
        # Residual blocks
        r1 = residual1d_block(inputs = u5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')
        r7 = residual1d_block(inputs = r6, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block7_')
        r8 = residual1d_block(inputs = r7, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block8_')

        # downsample
        d1 = downsample1d_block(inputs = r8, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block1_')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block2_')
        d3 = downsample1d_block(inputs = d2, filters = 256, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block3_')
        d4 = downsample1d_block(inputs = d3, filters = 128, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block4_')
        d5 = downsample1d_block(inputs = d4, filters = 64, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block5_')
        # Output
        o1 = conv1d_layer(inputs = d5, filters = 24, kernel_size = 5, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')
        
    return o2
    

def generator_beta2alpha(inputs, reuse = False, scope_name = 'generator_beta2alpha'):
   
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose_1')
   
    with tf.variable_scope(scope_name) as scope:
       
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h0 = conv1d_layer(inputs = inputs, filters = 24, kernel_size = 5, strides = 1, activation = None , name = 'h0_conv1')
        h0_gates = conv1d_layer(inputs = inputs, filters = 24, kernel_size = 5, strides = 1, activation = None, name = 'h0_conv_gates1')
        h0_glu = gated_linear_layer(inputs = h0, gates = h0_gates, name = 'h0_glu1')
       
        h1 = conv1d_layer(inputs = h0_glu, filters = 64, kernel_size = 5, strides = 1, activation = None , name = 'h1_conv1')
        h1_gates = conv1d_layer(inputs = h0_glu, filters = 64, kernel_size = 5, strides = 1, activation = None, name = 'h1_conv_gates1')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu1')
        

        # upsample
        u1 = upsample1d_block(inputs = h1_glu, filters = 64, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block1_')
        u2 = upsample1d_block(inputs = u1, filters = 128, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block2_1')
        u3 = upsample1d_block(inputs = u2, filters = 256, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block3_1')
        u4 = upsample1d_block(inputs = u3, filters = 512, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block4_1')
        u5 = upsample1d_block(inputs = u4, filters = 1024, kernel_size = 5, strides = 1, name_prefix = 'upsample1d_block5_1')
        
        # Residual blocks
        r1 = residual1d_block(inputs = u5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_1')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_1')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_1')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_1')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_1')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_1')
        r7 = residual1d_block(inputs = r6, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block7_1')
        r8 = residual1d_block(inputs = r7, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block8_1')

        # downsample
        d1 = downsample1d_block(inputs = r8, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block1_1')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block2_1')
        d3 = downsample1d_block(inputs = d2, filters = 256, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block3_1')
        d4 = downsample1d_block(inputs = d3, filters = 128, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block4_1')
        d5 = downsample1d_block(inputs = d4, filters = 64, kernel_size = 5, strides = 1, shuffle_size = 1, name_prefix = 'downsample1d_block5_1')
        # Output
        o1 = conv1d_layer(inputs = d5, filters = 24, kernel_size = 5, strides = 1, activation = None, name = 'o1_conv1')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose1')
        
    return o2
    
def discriminator_Sig(inputs, reuse = False, scope_name = 'discriminator_Sig'):

    
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
       
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        h0 = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_Sig')
        h0_gates = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_gates_Sig')
        h0_glu = gated_linear_layer(inputs = h0, gates = h0_gates, name = 'h0_glu_Sig')

 
        h1 = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_Sig')
        h1_gates = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_gates_Sig')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu_Sig')

       
        d0 = upsample2d_block(inputs = h1_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block0__Sig')
        d1 = upsample2d_block(inputs = d0, filters = 128, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block1__Sig')
        dx = upsample2d_block(inputs = d1, filters = 256, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_blockx__Sig')
        
        d2 = upsample2d_block(inputs = dx, filters = 512, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block2__Sig')
        d3 = upsample2d_block(inputs = d2, filters = 1024, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block3__Sig')

       
        Sig1 = tf.layers.dense(inputs = d3, units = 1, activation = tf.nn.sigmoid)
        
       
        
        return Sig1


def discriminator_Rel(inputs, reuse = False, scope_name = 'discriminator_Rel'):

    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        h0 = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_Rel')
        h0_gates = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_gates_Rel')
        h0_glu = gated_linear_layer(inputs = h0, gates = h0_gates, name = 'h0_glu_Rel')

 
        h1 = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_Rel')
        h1_gates = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_gates_Rel')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu_Rel')

        d0 = upsample2d_block(inputs = h1_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block0_Rel')
        dx = upsample2d_block(inputs = d0, filters = 128, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_blockx_Rel')
        d1 = upsample2d_block(inputs = dx, filters = 256, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block1_Rel')
        d2 = upsample2d_block(inputs = d1, filters = 512, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block2_Rel')
        d3 = upsample2d_block(inputs = d2, filters = 1024, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block3_Rel')

        
        Rel2 = tf.layers.dense(inputs = d3, units = 1, activation = tf.nn.relu)
        
        return Rel2






def discriminator_Sel(inputs, reuse = False, scope_name = 'discriminator_Sel'):

   
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        h0 = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_Sel')
        h0_gates = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_gates_Sel')
        h0_glu = gated_linear_layer(inputs = h0, gates = h0_gates, name = 'h0_glu_Sel')

 
        h1 = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_Sel')
        h1_gates = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_gates_Sel')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu_Sel')

       
        d0 = upsample2d_block(inputs = h1_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block0_Sel')
        dx = upsample2d_block(inputs = d0, filters = 128, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_blockx_Sel')
        d1 = upsample2d_block(inputs = dx, filters = 256, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block1_Sel')
        d2 = upsample2d_block(inputs = d1, filters = 512, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block2_Sel')
        d3 = upsample2d_block(inputs = d2, filters = 1024, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block3_Sel')

        
        Sel3=tf.layers.dense(inputs = d3, units = 1, activation = tf.nn.selu)
       
        
        return Sel3


def discriminator_Elu(inputs, reuse = False, scope_name = 'discriminator_Elu'):

   
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        h0 = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_Elu')
        h0_gates = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_gates_Elu')
        h0_glu = gated_linear_layer(inputs = h0, gates = h0_gates, name = 'h0_glu_Elu')

 
        h1 = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_Elu')
        h1_gates = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_gates_Elu')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu_Elu')

        
        d0 = upsample2d_block(inputs = h1_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block0_Elu')
        dx = upsample2d_block(inputs = d0, filters = 128, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_blockx_Elu')
        d1 = upsample2d_block(inputs = dx, filters = 256, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block1_Elu')
        d2 = upsample2d_block(inputs = d1, filters = 512, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block2_Elu')
        d3 = upsample2d_block(inputs = d2, filters = 1024, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block3_Elu')

        
        Elu4=tf.layers.dense(inputs = d3, units = 1, activation = tf.nn.elu)
        
        
        return Elu4


def discriminator_LRel(inputs, reuse = False, scope_name = 'discriminator_LRel'):

   
    inputs = tf.expand_dims(inputs, -1)

    with tf.variable_scope(scope_name) as scope:
        
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False
        h0 = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_LRel')
        h0_gates = conv2d_layer(inputs = inputs, filters = 24, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h0_conv_gates_LRel')
        h0_glu = gated_linear_layer(inputs = h0, gates = h0_gates, name = 'h0_glu_LRel')

 
        h1 = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_LRel')
        h1_gates = conv2d_layer(inputs =  h0_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], activation = None, name = 'h1_conv_gates_LRel')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu_LRel')

        d0 = upsample2d_block(inputs = h1_glu, filters = 64, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block0_LRel')
        dx = upsample2d_block(inputs = d0, filters = 128, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_blockx_LRel')
        d1 = upsample2d_block(inputs = dx, filters = 256, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block1_LRel')
        d2 = upsample2d_block(inputs = d1, filters = 512, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block2_LRel')
        d3 = upsample2d_block(inputs = d2, filters = 1024, kernel_size = [4, 4], strides = [1, 2], name_prefix = 'upsample2d_block3_LRel')

        
        Lrel=tf.layers.dense(inputs = d3, units = 1, activation = partial(tf.nn.leaky_relu, alpha=0.01))
        
        
        return Lrel



