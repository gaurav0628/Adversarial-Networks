
"""
Created on Fri Dec 1 18:48:52 2020

@author: gauravthapliyal
"""

from scipy import misc
import matplotlib.pyplot as plt 
import config
import logging
import PIL,os,sys
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import random
import tensorflow_probability as tfp

data_source = '/Users/gauravthapliyal/Fall_2020/Machine_Learning/VAE_GAN/img_align_celeba/img_align_celeba/'

crop_style = 'closecrop'
encoder_learning_rate = 0.0003;
decoder_learning_rate = 0.0003;
discriminator_learning_rate = 0.0001;

batch_size = 128
n_epoch = 50
z_dim = 128

img_height = 64
img_width = 64
num_channels = 3
n_inputs = 64*64
n_outputs = 10

X = tf.placeholder(tf.float32,[None,img_height,img_width,num_channels]);
epoch_number = tf.placeholder(tf.float32,[]);

stddev = 0.02

def discriminator(X,isTrainable=True,reuse=False,name='discriminator'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1')
        conv1 = tf.nn.relu(conv1,name='leaky_relu_conv_1')

        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2')
        conv2 = tf.nn.relu(conv2,name='leaky_relu_conv_2')
        
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3')
        conv3 = tf.nn.relu(conv3,name='leaky_relu_conv_3')
        
        conv4 = tf.layers.conv2d(conv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=512,kernel_size=[5,5],padding='SAME',strides=(1,1),name='enc_conv4_layer',activation=None,trainable=isTrainable,reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4,training=isTrainable,reuse=reuse,name='bn_4')
        conv4 = tf.nn.relu(conv4,name='leaky_relu_conv_4')
        
        conv4_flattened = tf.layers.flatten(conv4)
        l_th_layer_representation = conv4_flattened

        output_disc = tf.layers.dense(conv4_flattened,1,activation=tf.nn.sigmoid,name='dis_fc_layer',trainable=isTrainable,reuse=reuse)
        return l_th_layer_representation,output_disc

def encoder(X,isTrainable=True,reuse=False,name='encoder'):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables();
        conv1 = tf.layers.conv2d(X,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv1_layer',activation=None,trainable=isTrainable,reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1,training=isTrainable,reuse=reuse,name='bn_1')
        conv1 = tf.nn.relu(conv1,name='leaky_relu_conv_1')

        conv2 = tf.layers.conv2d(conv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv2_layer',activation=None,trainable=isTrainable,reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2,training=isTrainable,reuse=reuse,name='bn_2')
        conv2 = tf.nn.relu(conv2,name='leaky_relu_conv_2')
        
        conv3 = tf.layers.conv2d(conv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',strides=(2,2),name='enc_conv3_layer',activation=None,trainable=isTrainable,reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3,training=isTrainable,reuse=reuse,name='bn_3')
        conv3 = tf.nn.relu(conv3,name='leaky_relu_conv_3')
        
        conv4 = tf.layers.conv2d(conv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=512,kernel_size=[5,5],padding='SAME',strides=(1,1),name='enc_conv4_layer',activation=None,trainable=isTrainable,reuse=reuse)
        conv4 = tf.layers.batch_normalization(conv4,training=isTrainable,reuse=reuse,name='bn_4')
        conv4 = tf.nn.relu(conv4,name='leaky_relu_conv_4')
        conv4_flattened = tf.layers.flatten(conv4)
        
        z_mean = tf.layers.dense(conv4_flattened,z_dim,name='enc_mean',trainable=isTrainable,reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        z_variance = tf.layers.dense(conv4_flattened,z_dim,activation=tf.nn.softplus,name='enc_variance',trainable=isTrainable,reuse=reuse,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        epsilon_val = epsilon_distribution(z_dim).sample(tf.shape(X)[0])
        z_sample = tf.add(z_mean,tf.multiply(z_variance,epsilon_val))

        dist = tfp.distributions.MultivariateNormalDiag(z_mean,z_variance)
        return dist,z_sample

def decoder(z_sample,isTrainable=True,reuse=False,name='decoder')
    with tf.variable_scope(name) as scope:  
        if reuse:
            scope.reuse_variables()

        z_sample = tf.layers.dense(z_sample,8*8*512,activation=None,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_first_layer',kernel_initializer=tf.truncated_normal_initializer(stddev=stddev))
        z_sample = tf.layers.batch_normalization(z_sample,training=isTrainable,reuse=reuse,name='bn_0')
        z_sample = tf.nn.relu(z_sample)
        z_sample = tf.reshape(z_sample,[-1,8,8,512])

        deconv1 = tf.layers.conv2d_transpose(z_sample,kernel_initializer=tf.random_normal_initializer(stddev=stddev),filters=256,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv1_layer',trainable=isTrainable,reuse=reuse)
        deconv1 = tf.layers.batch_normalization(deconv1,training=isTrainable,reuse=reuse,name='bn_1')
        deconv1 = tf.nn.relu(deconv1,name='relu_deconv_1')
         
        deconv2 = tf.layers.conv2d_transpose(deconv1,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=128,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv2_layer',trainable=isTrainable,reuse=reuse)
        deconv2 = tf.layers.batch_normalization(deconv2,training=isTrainable,reuse=reuse,name='bn_2')
        deconv2 = tf.nn.relu(deconv2,name='relu_deconv_2')
        
        deconv3 = tf.layers.conv2d_transpose(deconv2,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=64,kernel_size=[5,5],padding='SAME',activation=None,strides=(2,2),name='dec_deconv3_layer',trainable=isTrainable,reuse=reuse)
        deconv3 = tf.layers.batch_normalization(deconv3,training=isTrainable,reuse=reuse,name='bn_3')
        deconv3 = tf.nn.relu(deconv3,name='relu_deconv_3')
        
        deconv4 = tf.layers.conv2d_transpose(deconv3,kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),filters=3,kernel_size=[5,5],padding='SAME',activation=None,strides=(1,1),name='dec_deconv4_layer',trainable=isTrainable,reuse=reuse)    
        deconv4 = tf.nn.tanh(deconv4)
        
        deconv_4_reshaped = tf.reshape(deconv4,[-1,img_height,img_width,num_channels])
        return deconv_4_reshaped

posterior_dist,z_sample = encoder(X)
prior_dist = prior_z(z_dim)
generated_sample = prior_dist.sample(batch_size)
reconstructed_x_tilde = decoder(z_sample)
test_reconstruction = decoder(z_sample,isTrainable=False,reuse=True)
reconstructed_x_dash = decoder(generated_sample,reuse=True)
true_x_l_th_layer_representation,Dis_X = discriminator(X)
x_tilde_l_th_layer_representation,Dis_x_tilde = discriminator(reconstructed_x_tilde,reuse=True)
x_dash_l_th_layer_representation,Dis_x_dash = discriminator(reconstructed_x_dash,reuse=True)
ae_loss = tf.reduce_mean(tf.pow(X- reconstructed_x_tilde,2))
gan_loss = tf.reduce_mean(tf.add(tf.add(tf.log(Dis_X),tf.log(1-Dis_x_tilde)),tf.log(1-Dis_x_dash)))
gan_loss = -1 * gan_loss
dis_l_layer_loss = tf.reduce_mean(tf.pow(x_tilde_l_th_layer_representation - true_x_l_th_layer_representation,2))
kl_loss = tf.reduce_mean(tfp.distributions.kl_divergence(posterior_dist,prior_dist))
encoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
decoder_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='decoder')
discriminator_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
gamma1 = 30
decoder_loss = gamma1*dis_l_layer_loss + tf.reduce_mean(- tf.log(Dis_x_tilde) - tf.log(Dis_x_dash))

discriminator_loss = gan_loss
kl_weightage = 1/(batch_size)

gamma2 = 10
encoder_loss = kl_weightage*kl_loss + gamma2*dis_l_layer_loss

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops)

    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001,beta1=0.5)
    autoencoder_gradsVars = autoencoder_optimizer.compute_gradients(ae_loss, encoder_params+decoder_params)
    autoencoder_train_optimizer = autoencoder_optimizer.apply_gradients(autoencoder_gradsVars)

    encoder_optimizer = tf.train.AdamOptimizer(learning_rate = encoder_learning_rate,beta1=0.5)
    encoder_gradsVars = encoder_optimizer.compute_gradients(encoder_loss, encoder_params)
    encoder_train_optimizer = encoder_optimizer.apply_gradients(encoder_gradsVars)

    decoder_optimizer = tf.train.AdamOptimizer(learning_rate = decoder_learning_rate,beta1=0.5,beta2=0.999)
    decoder_gradsVars = decoder_optimizer.compute_gradients(decoder_loss, decoder_params)
    decoder_train_optimizer = decoder_optimizer.apply_gradients(decoder_gradsVars)

    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = discriminator_learning_rate,beta1=0.5)
    discriminator_gradsVars = discriminator_optimizer.compute_gradients(discriminator_loss, discriminator_params)
    discriminator_train_optimizer = discriminator_optimizer.apply_gradients(discriminator_gradsVars)

tf.summary.scalar("kl_loss ",kl_weightage*kl_loss)
tf.summary.scalar("Discriminator_Lth_layer_loss in Encoder ",gamma2*dis_l_layer_loss)
tf.summary.scalar("Discriminator_Lth_layer_loss in Decoder ",gamma1*dis_l_layer_loss)
tf.summary.scalar("encoder_loss",encoder_loss)
tf.summary.scalar("decoder_loss",decoder_loss)
tf.summary.scalar("discriminator_loss",discriminator_loss)

for g,v in encoder_gradsVars:    
    tf.summary.histogram(v.name,v)
    tf.summary.histogram(v.name+str('grad'),g)

for g,v in decoder_gradsVars:    
    tf.summary.histogram(v.name,v)
    tf.summary.histogram(v.name+str('grad'),g)

for g,v in discriminator_gradsVars:    
    tf.summary.histogram(v.name,v)
    tf.summary.histogram(v.name+str('grad'),g)

merged_all = tf.summary.merge_all()
log_directory = 'VAE-GAN-dir'
model_directory='VAE-GAN-model_dir'

def train():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mode = 'train'
        train_min_file_num = 1
        train_max_file_num = 162770
        train_files = range(train_min_file_num, 1+train_max_file_num);
        train_file_iter=[os.path.join(data_source, '%s' % str(i).zfill(6)) for i in train_files]
        val_min_file_num = 1
        val_max_file_num = 162770
        val_files = range(val_min_file_num, 1+val_max_file_num)
        val_file_iter=[os.path.join(data_source, '%s' % str(i).zfill(6)) for i in val_files]
        n_batches = 28
        n_batches = int(n_batches)
        print('n_batches : ',n_batches,' when batch_size : ',batch_size)
        temp_batch = 1
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(log_directory,sess.graph)
        iterations = 0

        for epoch in range(n_epoch):
            for batch in range(n_batches):
                iterations += 1
                
                k=1;
                for i in range(k):
                    X_batch = get_random_batch(train_file_iter,batch_size)
                    fd = {X:X_batch,epoch_number:epoch+1}
                    _,dis_loss= sess.run([discriminator_train_optimizer,discriminator_loss],feed_dict = fd)

                j=1;
                for i in range(j):
                    X_batch = get_random_batch(train_file_iter,batch_size)
                    fd = {X:X_batch,epoch_number:epoch+1}
                    _,enc_loss = sess.run([encoder_train_optimizer,encoder_loss],feed_dict = fd)

                m=1;
                for i in range(m):
                    X_batch = get_random_batch(train_file_iter,batch_size)
                    fd = {X:X_batch,epoch_number:epoch+1}
                    _,dec_loss,kl_div_loss,merged = sess.run([decoder_train_optimizer,decoder_loss,kl_loss,merged_all],feed_dict = fd)
                
                if(iterations%20==0):
                    writer.add_summary(merged,iterations)

            if(epoch%2==0):
                num_val_img = 25
                batch_X = get_random_batch(val_file_iter,num_val_img)
                recons = sess.run(test_reconstruction,feed_dict={X:batch_X,epoch_number:1+epoch})
                recons = np.reshape(recons,[-1,64,64,3])
                n_gen = 25;
                sample = tf.random_normal([n_gen,z_dim])
                generations = sess.run(test_reconstruction,feed_dict={z_sample:sample.eval(),epoch_number:1+epoch})
                generations = np.reshape(generations,[-1,64,64,3])

                temp_index = -1;
                for s in range(generations.shape[0]):
                    temp_index += 1
                    generations[temp_index] = denormalize_image(generations[temp_index])

                temp_index = -1;
                for s in range(batch_X.shape[0]):
                    temp_index += 1
                    batch_X[temp_index] = denormalize_image(batch_X[temp_index])

                temp_index = -1;
                for s in range(recons.shape[0]):
                    temp_index += 1
                    recons[temp_index] = denormalize_image(recons[temp_index])

                n = 5
                reconstructed = np.empty((64*n,64*n,3))
                original = np.empty((64*n,64*n,3))
                generated_images = np.empty((64*n,64*n,3))
                for i in range(n):
                    for j in range(n):
                        original[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = batch_X[i*n+j]
                        reconstructed[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = recons[i*n+j]
                        generated_images[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64,:] = generations[i*n+j]
                plt.figure(figsize=(n, n))
                plt.imshow(original, origin="upper",interpolation='nearest', cmap="gray")
                plt.savefig('op-real/'+str(epoch)+'.png')
                plt.close()
                plt.figure(figsize=(n, n))
                plt.imshow(reconstructed, origin="upper",interpolation='nearest', cmap="gray")
                plt.savefig('op-recons/'+str(epoch)+'.png')
                plt.close()
                plt.figure(figsize=(n, n))
                plt.imshow(generated_images, origin="upper",interpolation='nearest', cmap="gray")
                plt.savefig('op-gen/'+str(epoch)+'.png')
                plt.close()

            if (epoch % 5) == 0:
                save_path = saver.save(sess, model_directory+'/model_'+str(epoch))


def test():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        saver = tf.train.Saver(var_list=params)

        for var in params:
            print (var.name+"\t")
        string = model_directory+'/model_'+str(50)
        try:
            saver.restore(sess, string)
        except:
            sys.exit(0)

        n = 5;
        
        reconstructed = np.empty((28*n,28*n))
        original = np.empty((28*n,28*n))

        for i in range(n):
            recons = sess.run(test_reconstruction,feed_dict={X:batch_X})
            print ('recons : ',recons.shape)
            recons = np.reshape(recons,[-1,784])
            print ('recons : ',recons.shape)
            for j in range(n):
                    original[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_X[j].reshape([28, 28])
            for j in range(n):
                reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28])

        plt.figure(figsize=(n, n));
        plt.imshow(original, origin="upper", cmap="gray")
        plt.savefig('original_new_vae.png')
        plt.figure(figsize=(n, n))
        plt.imshow(reconstructed, origin="upper", cmap="gray")
        plt.savefig('reconstructed_new_vae.png')

        n=15
        reconstructed = np.empty((28*n,28*n))
        for i in range(n):
            sample = tf.random_normal([n,z_dim])
            recons = sess.run(test_reconstruction,feed_dict={z_sample:sample.eval()})
            recons = np.reshape(recons,[-1,784])

            for j in range(n):
                reconstructed[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons[j].reshape([28, 28])

        plt.figure(figsize=(n, n))
        plt.imshow(reconstructed, origin="upper", cmap="gray")
        plt.title('Generated Image')
        plt.savefig('gen-img.png')
        plt.close()

#Utility functions

def denormalize_image(image):
    image /= 2. 
    image = image + 0.5 
    return image;


def crop(im):
    width = 178
    height = 218
    new_width = 140
    new_height = 140
    if crop_style == 'closecrop':
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height)/2
        im = im.crop((left, top, right, bottom))
        im = im.resize((64, 64), PIL.Image.ANTIALIAS)
    elif self.crop_style == 'resizecrop':
        im = im.resize((64, 78), PIL.Image.ANTIALIAS)
        im = im.crop((0, 7, 64, 64 + 7))
    return np.array(im).reshape(64, 64, 3) / 255.



def normalize_image(image):
    normalized_image = image - 0.5;
    normalized_image *= 2;
    return normalized_image

def get_random_batch(file_iter,batch_size = 3,):
    random_file_iter = np.random.choice(file_iter,batch_size,replace=False)
    X = np.zeros([len(random_file_iter),img_height,img_width, num_channels])
    index = -1
    for f in random_file_iter:
        index += 1
        f = f + '.jpg'
        print(f)
        curr_img = Image.open(f)
        curr_img = crop(curr_img)
        curr_img = normalize_image(curr_img)
        X[index] = curr_img
    return X


def prior_z(latent_dim):
    z_mean = tf.zeros(latent_dim)
    z_var = tf.ones(latent_dim)
    return tfp.distributions.MultivariateNormalDiag(z_mean,z_var)

def epsilon_distribution(latent_dim)
    eps_mean = tf.zeros(latent_dim)
    eps_var = tf.ones(latent_dim)
    return tfp.distributions.MultivariateNormalDiag(eps_mean,eps_var)

train();
test();






