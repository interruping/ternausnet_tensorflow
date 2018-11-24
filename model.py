import tensorflow as tf
import numpy as np
from torchvision import models
from tqdm import tqdm, trange
from batchup import data_source

from metrics import tf_iou_metric, lovasz_loss

conv2d = tf.layers.conv2d
upconv2d = tf.layers.conv2d_transpose
maxpool2d = tf.layers.max_pooling2d
dropout = tf.nn.dropout
concat = tf.concat

class Unet11(object):
    def __init__(self, sess, img_size):
        self._sess = sess

        self._global_step = tf.Variable(0, name='global_step', trainable=True)

        self._input = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 3), name='imgs_input')
        self._labels = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 1), name='imgs_label')
        
        self._learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self._dropout_rate = tf.placeholder(tf.float32, shape=(), name='dropout_rate')        
        
        encoded = self._encoding_layers(self._input, self._dropout_rate)
        output = self._decoding_layers(encoded, self._dropout_rate)
        
        self._y_logits_op = tf.nn.sigmoid(output, name='y_logits_op')
        
        self._loss_op = lovasz_loss(self._labels, output)
        self._loss_summary = tf.summary.scalar('loss', self._loss_op)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self._train_op = optimizer.minimize(loss=self._loss_op, global_step=self._global_step)
        self._eval_op = tf_iou_metric(label=self._labels, pred=self._y_logits_op)
        
        self._eval_summary = tf.summary.scalar('iou_metric', self._eval_op)

        self._train_summary_writer = tf.summary.FileWriter('./logs/tain', graph_def=sess.graph_def)
        self._validation_summary_writer =  tf.summary.FileWriter('./logs/validation', graph_def=sess.graph_def)

        self._hook_every_n = 0
        self._hook = None

        self._saver = tf.train.Saver(max_to_keep=100)

    def _get_pretrained_vgg11_weights(self):
        pretrained_vgg11 = models.vgg11(pretrained=True).features

        pretrained_weight = {}
        pretrained_bias = {}

        for i, x_layer in enumerate(pretrained_vgg11):
            if 'weight' in dir(x_layer):
                weights = x_layer.weight.detach().numpy()
                pretrained_weight[i] = weights
            else:
                pretrained_weight[i] = None

            if 'bias' in dir(x_layer):
                bias = x_layer.bias.detach().numpy()
                pretrained_bias[i] = bias
            else:
                pretrained_bias[i] = None

        return pretrained_weight, pretrained_bias  

    
        
    def _encoding_layers(self, input_imgs, dropout_rate):

        vgg11_w, vgg11_b = self._get_pretrained_vgg11_weights()

        vgg11_w = {
            'conv3_64': vgg11_w[0],
            'conv3_128': vgg11_w[3],
            'conv3_256_1': vgg11_w[6],
            'conv3_256_2': vgg11_w[8],
            'conv3_512_1': vgg11_w[11],
            'conv3_512_2': vgg11_w[13],
            'conv3_512_3': vgg11_w[16],
            'conv3_512_4': vgg11_w[18], 
        }

        vgg11_b =  {
            'conv3_64': vgg11_b[0],
            'conv3_128': vgg11_b[3],
            'conv3_256_1': vgg11_b[6],
            'conv3_256_2': vgg11_b[8],
            'conv3_512_1': vgg11_b[11],
            'conv3_512_2': vgg11_b[13],
            'conv3_512_3': vgg11_b[16],
            'conv3_512_4': vgg11_b[18], 
        }

        conv3_64_w_init = tf.constant_initializer(np.transpose(vgg11_w['conv3_64'], (1, 2, 3, 0)))
        conv3_64_b_init = tf.constant_initializer(vgg11_b['conv3_64'])

        cnc64 = conv2d(input_imgs, 64, [3,3], strides=1, padding='SAME', kernel_initializer=conv3_64_w_init, bias_initializer=conv3_64_b_init ,trainable=False, activation=tf.nn.relu)
        conv = maxpool2d(cnc64, [2, 2], strides=2)
        conv = dropout(conv, keep_prob=dropout_rate)

        conv3_128_w_init = tf.constant_initializer(np.transpose(vgg11_w['conv3_128'], (1, 2, 3, 0)))
        conv3_128_b_init = tf.constant_initializer(vgg11_b['conv3_128'])

        cnc128 = conv2d(conv, 128, [3,3], strides=1, padding='SAME', kernel_initializer=conv3_128_w_init, bias_initializer=conv3_128_b_init ,trainable=False, activation=tf.nn.relu)
        conv = maxpool2d(cnc128, [2, 2], strides=2)
        conv = dropout(conv, keep_prob=dropout_rate)

        conv3_256_1_w_init = tf.constant_initializer(np.transpose(vgg11_w['conv3_256_1'], (1, 2, 3, 0)))
        conv3_256_1_b_init = tf.constant_initializer(vgg11_b['conv3_256_1'])

        conv3_256_2_w_init= tf.constant_initializer(np.transpose(vgg11_w['conv3_256_2'], (1, 2, 3, 0)))
        conv3_256_2_b_init = tf.constant_initializer(vgg11_b['conv3_256_2'])

        conv = conv2d(conv, 256, [3,3], strides=1, padding='SAME', kernel_initializer=conv3_256_1_w_init, bias_initializer=conv3_256_1_b_init ,trainable=False, activation=tf.nn.relu)
        cnc256 = conv2d(conv, 256, [3,3], strides=1, padding='SAME', kernel_initializer=conv3_256_2_w_init, bias_initializer=conv3_256_2_b_init ,trainable=False, activation=tf.nn.relu)
        conv = maxpool2d(cnc256, [2, 2], strides=2)
        conv = dropout(conv, keep_prob=dropout_rate)

        conv3_512_1_w_init = tf.constant_initializer(np.transpose(vgg11_w['conv3_512_1'], (1, 2, 3, 0)))
        conv3_512_1_b_init = tf.constant_initializer(vgg11_b['conv3_512_1'])

        conv3_512_2_w_init = tf.constant_initializer(np.transpose(vgg11_w['conv3_512_2'], (1, 2, 3, 0)))
        conv3_512_2_b_init = tf.constant_initializer(vgg11_b['conv3_512_2'])

        conv = conv2d(conv, 512, [3,3], strides=1, padding='SAME', kernel_initializer=conv3_512_1_w_init, bias_initializer=conv3_512_1_b_init ,trainable=False, activation=tf.nn.relu)
        cnc512_1 = conv2d(conv, 512, [3,3], strides=1, padding='SAME', kernel_initializer=conv3_512_2_w_init, bias_initializer=conv3_512_2_b_init ,trainable=False, activation=tf.nn.relu)
        conv = maxpool2d(cnc512_1, [2, 2], strides=2)
        conv = dropout(conv, keep_prob=dropout_rate)

        conv3_512_3_w_init = tf.constant_initializer(np.transpose(vgg11_w['conv3_512_3'], (1, 2, 3, 0)))
        conv3_512_3_b_init = tf.constant_initializer(vgg11_b['conv3_512_3'])

        conv3_512_4_w_init = tf.constant_initializer(np.transpose(vgg11_w['conv3_512_4'], (1, 2, 3, 0)))
        conv3_512_4_b_init = tf.constant_initializer(vgg11_b['conv3_512_4'])

        conv = conv2d(conv, 512, [3,3], strides=1, padding='SAME', kernel_initializer=conv3_512_3_w_init, bias_initializer=conv3_512_3_b_init ,trainable=False, activation=tf.nn.relu)
        cnc512_2 = conv2d(conv, 512, [3,3], strides=1, padding='SAME', kernel_initializer=conv3_512_4_w_init, bias_initializer=conv3_512_4_b_init ,trainable=False, activation=tf.nn.relu)
        conv = maxpool2d(cnc512_2, [2, 2], strides=2)
        conv = dropout(conv, keep_prob=dropout_rate)

        result = {}
        result['conv'] = conv
        result['cnc64'] = cnc64
        result['cnc128'] = cnc128
        result['cnc256'] = cnc256
        result['cnc512_1'] = cnc512_1
        result['cnc512_2'] = cnc512_2

        return result
 
    def _decoding_layers(self, encoded, dropout_rate):
        conv = encoded['conv'] 
        cnc64 = encoded['cnc64'] 
        cnc128 = encoded['cnc128'] 
        cnc256 = encoded['cnc256'] 
        cnc512_1 = encoded['cnc512_1']
        cnc512_2 = encoded['cnc512_2'] 

        xv_initalizer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)

        conv = conv2d(conv, 512, [3, 3], strides=1, padding='SAME', kernel_initializer=xv_initalizer, trainable=True, activation=tf.nn.relu)

        t_conv = upconv2d(conv, 256, [3, 3], strides=2, padding='SAME', kernel_initializer=xv_initalizer)
        t_conv = concat([t_conv, cnc512_2], 3)
        t_conv = conv2d(t_conv, 512, [3, 3], strides=1, padding='SAME', kernel_initializer=xv_initalizer, trainable=True, activation=tf.nn.relu)
        t_conv = dropout(t_conv, keep_prob=dropout_rate)

        t_conv = upconv2d(t_conv, 256, [3, 3], strides=2, padding='SAME', kernel_initializer=xv_initalizer)
        t_conv = concat([t_conv, cnc512_1], 3)
        t_conv = conv2d(t_conv, 512, [3, 3], strides=1, padding='SAME', kernel_initializer=xv_initalizer, trainable=True, activation=tf.nn.relu)
        t_conv = dropout(t_conv, keep_prob=dropout_rate)


        t_conv = upconv2d(t_conv, 128, [3, 3], strides=2, padding='SAME', kernel_initializer=xv_initalizer)
        t_conv = concat([t_conv, cnc256], 3)
        t_conv = conv2d(t_conv, 256, [3, 3], strides=1, padding='SAME', kernel_initializer=xv_initalizer, trainable=True, activation=tf.nn.relu)
        t_conv = dropout(t_conv, keep_prob=dropout_rate)


        t_conv = upconv2d(t_conv, 64, [3, 3], strides=2, padding='SAME', kernel_initializer=xv_initalizer)
        t_conv = concat([t_conv, cnc128], 3)
        t_conv = conv2d(t_conv, 128, [3, 3], strides=1, padding='SAME', kernel_initializer=xv_initalizer, trainable=True, activation=tf.nn.relu)
        t_conv = dropout(t_conv, keep_prob=dropout_rate)


        t_conv = upconv2d(t_conv, 32, [3, 3], strides=2, padding='SAME', kernel_initializer=xv_initalizer)
        t_conv = concat([t_conv, cnc64], 3)
        output = conv2d(t_conv, 1, [3, 3], strides=1, padding='SAME', kernel_initializer=xv_initalizer, trainable=True)

        return output

    def set_train_hook(self, every_n, hook):
        self._hook = hook
        self._hook_every_n = every_n

    def train(self, input_imgs, label_imgs, batch_size, dropout_rate, learning_rate):
        train_ds = data_source.ArrayDataSource([input_imgs, label_imgs])
        train_batch_iter = train_ds.batch_iterator(batch_size=batch_size)

        train_batch_total = len(input_imgs) // batch_size if len(input_imgs) % batch_size == 0 else len(input_imgs) // batch_size + 1
        
        epoch_train_loss = []
        epoch_train_iou = []

        train_batch_tqdm = tqdm(train_batch_iter, total=train_batch_total)
        for train_step, [minibatch_train_imgs, minibatch_train_labels] in enumerate(train_batch_tqdm):
            train_batch_tqdm.set_description('training...')

            train_feed = {
                self._input: minibatch_train_imgs,
                self._labels: minibatch_train_labels,
                self._dropout_rate: dropout_rate,
                self._learning_rate: learning_rate
            }

            minibatch_loss, loss_summary, _ = \
            self._sess.run([self._loss_op, self._loss_summary , self._train_op], feed_dict=train_feed)

            eval_feed = {
                self._input: minibatch_train_imgs,
                self._labels: minibatch_train_labels,
                self._dropout_rate: 1.0,
            }

            minibatch_eval, eval_summary = self._sess.run([self._eval_op, self._eval_summary], feed_dict=train_feed)

            epoch_train_loss.append(minibatch_loss)
            epoch_train_iou.append(minibatch_eval)

            global_step = self._sess.run(self._global_step)
            
            self._train_summary_writer.add_summary(loss_summary, global_step=global_step)
            self._train_summary_writer.flush()
            self._train_summary_writer.add_summary(eval_summary, global_step=global_step)
            self._train_summary_writer.flush()


            train_batch_tqdm.set_postfix(minibatch_loss=minibatch_loss, minibatch_iou=minibatch_eval)

            if self._hook and global_step % self._hook_every_n == 0:
                self._hook()
        
        epoch_train_loss = np.mean(epoch_train_loss)
        epoch_train_iou = np.mean(epoch_train_iou)

        return epoch_train_loss, epoch_train_iou

    def validate(self, input_imgs, label_imgs, batch_size):
        valid_ds = data_source.ArrayDataSource([input_imgs, label_imgs])
        valid_batch_iter = valid_ds.batch_iterator(batch_size=batch_size)

        valid_batch_total = len(input_imgs) // batch_size if len(input_imgs) % batch_size == 0 else len(input_imgs) // batch_size + 1
        
        epoch_valid_loss = []
        epoch_valid_iou = []

        valid_batch_tqdm = tqdm(valid_batch_iter, total=valid_batch_total)
        for valid_step, [minibatch_valid_imgs, minibatch_valid_labels] in enumerate(valid_batch_tqdm):
            valid_batch_tqdm.set_description('validating...')

            train_feed = {
                self._input: minibatch_valid_imgs,
                self._labels: minibatch_valid_labels,
                self._dropout_rate: 1.0,
            }

            minibatch_loss, minibatch_eval = \
            self._sess.run([self._loss_op, self._eval_op], feed_dict=train_feed)
            
            epoch_valid_loss.append(minibatch_loss)
            epoch_valid_iou.append(minibatch_eval)

            valid_batch_tqdm.set_postfix(minibatch_loss=minibatch_loss, minibatch_iou=minibatch_eval)
    
        epoch_valid_loss = np.mean(epoch_valid_loss)
        epoch_valid_iou = np.mean(epoch_valid_iou)
        
        global_step = self._sess.run(self._global_step)

        valid_loss_summary = tf.Summary()
        valid_loss_summary.value.add(tag="loss", simple_value=epoch_valid_loss)
        valid_loss_summary.value.add(tag="iou_metric", simple_value=epoch_valid_iou)
        self._validation_summary_writer.add_summary(valid_loss_summary, global_step=global_step)
        self._validation_summary_writer.flush()

        return epoch_valid_loss, epoch_valid_iou

    def predict(self, input_imgs, batch_size):
        predict_ds = data_source.ArrayDataSource([input_imgs])
        predict_batch_iter = predict_ds.batch_iterator(batch_size=batch_size)

        predict_batch_total = len(input_imgs) // batch_size if len(input_imgs) % batch_size == 0 else len(input_imgs) // batch_size + 1
        
        epoch_logits = []

        predict_batch_tqdm = tqdm(predict_batch_iter, total=predict_batch_total)
        for predict_step , [minibatch_valid_imgs] in enumerate(predict_batch_tqdm):
            predict_batch_tqdm.set_description('predicting...')

            train_feed = {
                self._input: minibatch_valid_imgs,
                self._dropout_rate: 1.0,
            }

            minibatch_logits =  self._sess.run(self._y_logits_op , feed_dict=train_feed)
            
            epoch_logits.append(minibatch_logits)
        return np.concatenate(epoch_logits, axis=0)
    
    def save(self):
        self._saver.save(self._sess, './model/unet11', global_step=self._global_step, write_meta_graph=False)

    def restore(self, checkpoint):
        self._saver.restore(self._sess, checkpoint)

    def restore_latest(self):
        latest_model = tf.train.latest_checkpoint('./model/')
        if not latest_model:
            self._sess.run(tf.global_variables_initializer())
        else:
            self._saver.restore(self._sess, latest_model)

if __name__ == '__main__':
    sess = tf.InteractiveSession()
    Unet11(sess, 128)