import os
import tensorflow as tf
from tfgan_network import discriminator_Sig, discriminator_Rel,discriminator_Sel,discriminator_Elu,discriminator_LRel,generator_alpha2beta,generator_beta2alpha
from loss import l1_loss_reconstruction, l1_loss_identity, combined_loss
from datetime import datetime


class TF_GAN_model(object):

    def __init__(self, num_features, discriminator_Sig = discriminator_Sig,discriminator_Rel=discriminator_Rel,discriminator_Sel=discriminator_Sel,discriminator_Elu=discriminator_Elu,
discriminator_LRel=discriminator_LRel, generator_alpha2beta = generator_alpha2beta,generator_beta2alpha=generator_beta2alpha, mode = 'train', log_dir = './tensorboard_log'):

        self.num_features = num_features
        self.input_shape = [None, num_features, None] 

        self.discriminator_Sig = discriminator_Sig
        self.discriminator_Rel = discriminator_Rel
        self.discriminator_Sel = discriminator_Sel
        self.discriminator_Elu =discriminator_Elu
        self.discriminator_LRel =discriminator_LRel
        
        self.generator_alpha2beta = generator_alpha2beta
        self.generator_beta2alpha = generator_beta2alpha

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        # in tensorflow 1x version
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries = self.summary()



#TF-GAN model details 

    def build_model(self):

        # Placeholders for real samples
        self.input_alpha_real = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_alpha_real')
        self.input_beta_real = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_beta_real')

        # Placeholders for generated samples
        self.input_alpha_fake = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_alpha_fake')
        self.input_beta_fake = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_beta_fake')

        # Placeholder for test samples
        self.input_alpha_test = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_alpha_test')
        self.input_beta_test = tf.placeholder(tf.float32, shape = self.input_shape, name = 'input_beta_test')

        #generated data
        self.generated_beta = self.generator_alpha2beta(inputs = self.input_alpha_real, reuse =True, scope_name = 'generator_alpha_to_beta')
        self.generated_alpha= self.generator_beta2alpha(inputs = self.input_beta_real, reuse = True, scope_name = 'generator_beta_to_alpha')
        
        #reconstructed data
        self.reconstructed_alpha = self.generator_beta2alpha(inputs =  self.generated_beta, reuse =True, scope_name ='generator_beta_to_alpha')
        self.reconstructed_beta= self.generator_alpha2beta(inputs = self.generated_alpha, reuse = True, scope_name = 'generator_alpha_to_beta')
        
        #identity traced data
        self.generated_alpha_identity = self.generator_beta2alpha(inputs = self.input_alpha_real, reuse = True, scope_name = 'generator_beta_to_alpha')
        self.generated_beta_identity = self.generator_alpha2beta(inputs = self.input_beta_real, reuse = True, scope_name = 'generator_alpha_to_beta')
        
        #discriminator (MCEPs using Sigmoid, ReLU, SeLU, ELU, Leaky ReLU)
        self.discriminator_alpha_fake_Sig = self.discriminator_Sig(inputs =self.generated_alpha, reuse = False, scope_name = 'discriminator_alpha_Sig')
        self.discriminator_alpha_fake_Rel = self.discriminator_Rel(inputs =self.generated_alpha, reuse = False, scope_name = 'discriminator_alpha_Rel')
        self.discriminator_alpha_fake_Sel = self.discriminator_Sel(inputs =self.generated_alpha, reuse = False, scope_name = 'discriminator_alpha_Sel')
        self.discriminator_alpha_fake_Elu = self.discriminator_Elu(inputs =self.generated_alpha, reuse = False, scope_name = 'discriminator_alpha_Elu')
        self.discriminator_alpha_fake_LRel = self.discriminator_LRel(inputs =self.generated_alpha, reuse = False, scope_name = 'discriminator_alpha_LReLU')

        
         self.discriminator_beta_fake_Sig = self.discriminator_Sig(inputs =self.generated_beta, reuse = False, scope_name = 'discriminator_beta_Sig')
        self.discriminator_beta_fake_Rel = self.discriminator_Rel(inputs =self.generated_beta, reuse = False, scope_name = 'discriminator_beta_Rel')
        self.discriminator_beta_fake_Sel = self.discriminator_Sel(inputs =self.generated_beta, reuse = False, scope_name = 'discriminator_beta_Sel')
        self.discriminator_beta_fake_Elu = self.discriminator_Elu(inputs =self.generated_beta, reuse = False, scope_name = 'discriminator_beta_Elu')
        self.discriminator_beta_fake_LRel = self.discriminator_LRel(inputs =self.generated_beta, reuse = False, scope_name = 'discriminator_beta_LReLU')

        

        
        
        # reconstruction loss
        self.reconstruction_loss = l1_loss_reconstruction(real = self.input_alpha_real, reconstructed = self.reconstructed_alpha) + l1_loss_reconstruction(real = self.input_beta_real,reconstructed = self.reconstructed_beta)

        # identity_tracing loss
        self.identity_tracing_loss = l1_loss_identity(real = self.input_alpha_real, identity = self.generated_alpha_identity) +  l1_loss_identity(real = self.input_beta_real, identity = self.generated_beta_identity)

        # Place holder for lambda_reconstruction  and lambda_identity_tracing
        self.lambda_reconstruction = tf.placeholder(tf.float32, None, name = 'lambda_reconstruction')
        self.lambda_identity_tracing = tf.placeholder(tf.float32, None, name = 'lambda_identity_tracing')

        # Generator loss
        # Generator wants to fool discriminator
        self.generator_loss_alpha2beta = combined_loss(target = tf.ones_like(self.discriminator_beta_fake_Sig) , sig = self.discriminator_beta_fake_Sig, rel =  self.discriminator_beta_fake_Rel , sel =self.discriminator_beta_fake_Sel , elu = self.discriminator_beta_fake_Elu , lrel = self.discriminator_beta_fake_LRel )

        self.generator_loss_beta2alpha = combined_loss(target = tf.ones_like(self.discriminator_alpha_fake_Sig) , sig = self.discriminator_alpha_fake_Sig, rel =  self.discriminator_alpha_fake_Rel , sel =self.discriminator_alpha_fake_Sel , elu = self.discriminator_alpha_fake_Elu , lrel = self.discriminator_alpha_fake_LRel)

        # total generator loss of TF-GAN
        self.tfgan_generator_loss =  (self.generator_loss_alpha2beta + self.generator_loss_beta2alpha)/2 + self.lambda_reconstruction*self.reconstruction_loss + self.lambda_identity_tracing*self.identity_tracing_loss


        
       

        self.discriminator_input_alpha_real_Sig = self.discriminator_Sig(inputs = self.self.input_alpha_real, reuse = True, scope_name = 'discriminator_alpha_Sig')
        self.discriminator_input_alpha_real_Rel = self.discriminator_Rel(inputs = self.self.input_alpha_real, reuse = True, scope_name = 'discriminator_alpha_Rel')
        self.discriminator_input_alpha_real_Sel = self.discriminator_Sel(inputs = self.self.input_alpha_real, reuse = True, scope_name = 'discriminator_alpha_Sel')
        self.discriminator_input_alpha_real_Elu = self.discriminator_Elu(inputs = self.self.input_alpha_real, reuse = True, scope_name = 'discriminator_alpha_Elu')
        self.discriminator_input_alpha_real_LRel = self.discriminator_LRel(inputs = self.self.input_alpha_real, reuse = True, scope_name = 'discriminator_alpha_LReLU')
       

        
        

        self.discriminator_input_beta_real_Sig = self.discriminator_Sig(inputs = self.self.input_beta_real, reuse = True, scope_name = 'discriminator_beta_Sig')
        self.discriminator_input_beta_real_Rel = self.discriminator_Rel(inputs = self.self.input_beta_real, reuse = True, scope_name = 'discriminator_beta_Rel')
        self.discriminator_input_beta_real_Sel = self.discriminator_Sel(inputs = self.self.input_beta_real, reuse = True, scope_name = 'discriminator_beta_Sel')
        self.discriminator_input_beta_real_Elu = self.discriminator_Elu(inputs = self.self.input_beta_real, reuse = True, scope_name = 'discriminator_beta_Elu')
        self.discriminator_input_beta_real_LRel = self.discriminator_LRel(inputs = self.self.input_beta_real, reuse = True, scope_name = 'discriminator_beta_LReLU')
       


        self.discriminator_input_alpha_fake_Sig = self.discriminator_Sig(inputs = self.input_alpha_fake, reuse = True, scope_name = 'discriminator_alpha_Sig')
        self.discriminator_input_alpha_fake_Rel = self.discriminator_Rel(inputs = self.input_alpha_fake, reuse = True, scope_name = 'discriminator_alpha_Rel')
        self.discriminator_input_alpha_fake_Sel = self.discriminator_Sel(inputs = self.input_alpha_fake, reuse = True, scope_name = 'discriminator_alpha_Sel')
        self.discriminator_input_alpha_fake_Elu = self.discriminator_Elu(inputs = self.input_alpha_fake, reuse = True, scope_name = 'discriminator_alpha_Elu')
        self.discriminator_input_alpha_fake_LRel = self.discriminator_LRel(inputs = self.input_alpha_fake, reuse = True, scope_name = 'discriminator_alpha_LReLU')
 
        

        self.discriminator_input_beta_fake_Sig = self.discriminator_Sig(inputs = self.input_beta_fake, reuse = True, scope_name = 'discriminator_beta_Sig')
        self.discriminator_input_beta_fake_Rel = self.discriminator_Rel(inputs = self.input_beta_fake, reuse = True, scope_name = 'discriminator_beta_Rel')
        self.discriminator_input_beta_fake_Sel = self.discriminator_Sel(inputs = self.input_beta_fake, reuse = True, scope_name = 'discriminator_beta_Sel')
        self.discriminator_input_beta_fake_Elu = self.discriminator_Elu(inputs = self.input_beta_fake, reuse = True, scope_name = 'discriminator_beta_Elu')
        self.discriminator_input_beta_fake_LRel = self.discriminator_LRel(inputs = self.input_beta_fake, reuse = True, scope_name = 'discriminator_beta_LReLU')
 

       
       
         self.discriminator_loss_input_alpha_real =  combined_loss(target = tf.ones_like(self.discriminator_input_alpha_real_Sig) , sig = self.discriminator_input_alpha_real_Sig , rel=  self.discriminator_input_alpha_real_Rel , sel = self.discriminator_input_alpha_real_Sel , elu= self.discriminator_input_alpha_real_Elu  , lrel= self.discriminator_input_alpha_real_LRel )

        self.discriminator_loss_input_alpha_fake =  combined_loss(target = tf.zeros_like(self.discriminator_input_alpha_fake_Sig), sig = self.discriminator_input_alpha_fake_Sig , rel=  self.discriminator_input_alpha_fake_Rel , sel = self.discriminator_input_alpha_fake_Sel , elu= self.discriminator_input_alpha_fake_Elu  , lrel= self.discriminator_input_alpha_fake_LRel )

        self.discriminator_loss_alpha = (self.discriminator_loss_input_alpha_real + self.discriminator_loss_input_alpha_fake) / 2

        self.discriminator_loss_input_beta_real =  combined_loss(target = tf.ones_like(self.discriminator_input_beta_real_Sig) , sig = self.discriminator_input_beta_real_Sig , rel=  self.discriminator_input_beta_real_Rel , sel = self.discriminator_input_beta_real_Sel , elu= self.discriminator_input_beta_real_Elu  , lrel= self.discriminator_input_beta_real_LRel )

        self.discriminator_loss_input_beta_fake =  combined_loss(target = tf.zeros_like(self.discriminator_input_beta_fake_Sig), sig = self.discriminator_input_beta_fake_Sig , rel=  self.discriminator_input_beta_fake_Rel , sel = self.discriminator_input_beta_fake_Sel , elu= self.discriminator_input_beta_fake_Elu  , lrel= self.discriminator_beta_alpha_fake_LRel )

        self.discriminator_loss_beta = (self.discriminator_loss_input_beta_real + self.discriminator_loss_input_beta_fake) / 2

        # merge two discriminator loss
        self.tfgan_discriminator_loss =  self.discriminator_loss_alpha + self.discriminator_loss_beta

        # Categorize variables 
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]
        
        # Reserved for test
        self.generator_beta_test = self.generator_alpha2beta(inputs = self.input_alpha_test, reuse = True, scope_name =
'generator_alpha_to_beta')
        self.generator_alpha_test = self.generator_beta2alpha(inputs = self.input_beta_test, reuse = True, scope_name = 'generator_beta_to_alpha')



#Adam optimizer

    def optimizer_initializer(self):

        self.generator_learning_rate = tf.placeholder(tf.float32, None, name = 'g_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name = 'd_learning_rate')
        self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = self.discriminator_learning_rate, beta1 = 0.5).minimize(self.tfgan_discriminator_loss, var_list = self.discriminator_vars)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate = self.generator_learning_rate, beta1 = 0.5).minimize(self.tfgan_generator_loss, var_list = self.generator_vars) 



    def train(self, input_A, input_B, lambda_reconstruction , lambda_identity_tracing, generator_learning_rate, discriminator_learning_rate):

        generated_alpha,generated_beta, tfgan_generator_loss, _, generator_summaries = self.sess.run(
            [self.generated_alpha, self.generated_beta, self.tfgan_generator_loss, self.generator_optimizer, self.generator_summaries], \
            feed_dict = {self.lambda_reconstruction : lambda_reconstruction , self.lambda_identity_tracing: lambda_identity_tracing, self.input_alpha_real: input_A, self.input_beta_real: input_B, self.generator_learning_rate: generator_learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        tfgan_discriminator_loss, _, discriminator_summaries = self.sess.run([self.tfgan_discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries], \
            feed_dict = {self.input_alpha_real: input_A, self.input_beta_real: input_B, self.discriminator_learning_rate: discriminator_learning_rate, self.input_alpha_fake: generated_alpha, self.input_beta_fake: generated_beta})

        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1

        return tfgan_generator_loss, tfgan_discriminator_loss


    def test(self, inputs, direction):

        if direction == 'alpha2beta':
            generation = self.sess.run(self.self.generator_beta_test, feed_dict = {self.input_alpha_test: inputs})
        elif direction == 'beta2alpha':
            generation = self.sess.run(self.self.generator_alpha_test, feed_dict = {self.input_beta_test: inputs})
        else:
            raise Exception('Conversion direction must be specified.')

        return generation


    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))
        
        return os.path.join(directory, filename)


    def load(self, filepath):

        self.saver.restore(self.sess, filepath)


    def summary(self):

        with tf.name_scope('generator_summaries'):
            reconstruction_loss_summary = tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
            identity_tracing_loss_summary = tf.summary.scalar('identity_tracing_loss', self.identity_tracing_loss)
            generator_loss_alpha2beta_summary = tf.summary.scalar('generator_loss_alpha2beta',self.generator_loss_alpha2beta )
            generator_loss_beta2alpha_summary = tf.summary.scalar('generator_loss_beta2alpha',  self.generator_loss_beta2alpha)
            tfgan_generator_loss_summary = tf.summary.scalar('tfgan_generator_loss', self.tfgan_generator_loss)
            generator_summaries = tf.summary.merge([reconstruction_loss_summary , identity_tracing_loss_summary, generator_loss_alpha2beta_summary, generator_loss_beta2alpha_summary, tfgan_generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_alpha_summary = tf.summary.scalar('discriminator_loss_alpha', self.discriminator_loss_alpha)
            discriminator_loss_beta_summary = tf.summary.scalar('discriminator_loss_beta',  self.discriminator_loss_beta)
            tfgan_discriminator_loss_summary = tf.summary.scalar('tfgan_discriminator_loss', self.tfgan_discriminator_loss)
            discriminator_summaries = tf.summary.merge([discriminator_loss_alpha_summary, discriminator_loss_beta_summary , tfgan_discriminator_loss_summary])

        return generator_summaries, discriminator_summaries


if __name__ == '__main__':
    
    model = TF_GAN_model(num_features = 24)
    
