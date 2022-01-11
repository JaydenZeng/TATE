import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np
from Settings import Config
from module import ff, multihead_attention, ln, mask, SigmoidAtt
import sys
from tensorflow.python.keras.utils import losses_utils



class MM:
    def __init__(self, is_training):
        self.config = Config()
        self.att_dim = self.config.att_dim
        self.visual = tf.placeholder(dtype = tf.float32, shape=[self.config.batch_size, self.config.max_visual_len, 709], name='visual')
        self.audio = tf.placeholder(dtype = tf.float32, shape = [self.config.batch_size, self.config.max_audio_len, 33], name='audio')
        self.text = tf.placeholder(dtype = tf.float32, shape = [self.config.batch_size, self.config.max_text_len, 768], name='text')
        self.label = tf.placeholder(dtype = tf.int32, shape = [self.config.batch_size], name = 'label')
        self.flag = tf.placeholder(dtype = tf.int32, shape = [self.config.batch_size], name = 'flag')
        self.pretrained_output = tf.placeholder(dtype = tf.float32, shape = [self.config.batch_size, 300, 304], name = 'pre')

        visual = tf.layers.dense(self.visual, self.config.att_dim, use_bias=False)
        audio = tf.layers.dense(self.audio, self.config.att_dim, use_bias=False)
        text = tf.layers.dense(self.text, self.config.att_dim, use_bias =False)
        
        with tf.variable_scope('vv', reuse=tf.AUTO_REUSE):
          enc_vv = multihead_attention(queries=visual,
                                   keys=visual,
                                   values=visual,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_vv = ff(enc_vv, num_units=[4*self.config.att_dim, self.config.att_dim])


        with tf.variable_scope('aa', reuse=tf.AUTO_REUSE):
          enc_aa = multihead_attention(queries=audio,
                                   keys=audio,
                                   values=audio,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_aa = ff(enc_aa, num_units=[4*self.config.att_dim, self.config.att_dim])

        with tf.variable_scope('tt', reuse=tf.AUTO_REUSE):
          enc_tt = multihead_attention(queries=text,
                                   keys=text,
                                   values=text,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_tt = ff(enc_tt, num_units=[4*self.config.att_dim, self.config.att_dim])

        with tf.variable_scope('all_weights', reuse = tf.AUTO_REUSE):
          Wr_wq = tf.get_variable('Wr_wq', [304, 1])
          Wm_wq = tf.get_variable('Wm_wq', [304, 304])
          Wu_wq = tf.get_variable('Wu_wq', [304, 304])

          Wr_wa = tf.get_variable('Wr_wa', [self.att_dim, 1])
          Wm_wa = tf.get_variable('Wm_wa', [self.att_dim, self.att_dim])
          Wu_wa = tf.get_variable('Wu_wa', [self.att_dim, self.att_dim])


          Wr_va = tf.get_variable('Wr_va', [self.att_dim, 1])
          Wm_va = tf.get_variable('Wm_va', [self.att_dim, self.att_dim])
          Wu_va = tf.get_variable('Wu_va', [self.att_dim, self.att_dim])

          wei_va = tf.get_variable('wei_va', [self.att_dim, 150])
          wei_vt = tf.get_variable('wei_vt', [self.att_dim, 150])
          wei_ta = tf.get_variable('wei_ta', [self.att_dim, 150])

          dis_va = tf.get_variable('wei_va', [self.att_dim, 150])
          dis_vt = tf.get_variable('wei_va', [self.att_dim, 150])
          dis_ta = tf.get_variable('wei_va', [self.att_dim, 150])

          W_l = tf.get_variable('W_l', [self.att_dim, self.config.class_num])
          b_l = tf.get_variable('b_l', [1, self.config.class_num])

        common_v = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_vv, [-1, self.att_dim]), wei_va), tf.matmul(tf.reshape(enc_vv, [-1, self.att_dim]), wei_vt)], -1), [self.config.batch_size, -1, self.att_dim])        
        common_a = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_aa, [-1, self.att_dim]), wei_va), tf.matmul(tf.reshape(enc_aa, [-1, self.att_dim]), wei_ta)], -1), [self.config.batch_size, -1, self.att_dim])
        common_t = tf.reshape(tf.concat([tf.matmul(tf.reshape(enc_tt, [-1, self.att_dim]), wei_vt), tf.matmul(tf.reshape(enc_tt, [-1, self.att_dim]), wei_ta)], -1), [self.config.batch_size, -1, self.att_dim])


        enc_all = tf.concat([common_v, common_a, common_t], 1)
                  
        # flag encoding
        enc_new = list(np.zeros([self.config.batch_size, 2]))
        for i in range(self.config.batch_size):
            if self.flag[i] == -1:
               enc_new[i] = tf.concat([enc_all[i], tf.tile([[1.0, 0.0, 0.0, 0.0]], [tf.shape(enc_all)[1], 1])], -1)
            elif self.flag[i] == 0:
                enc_new[i] = tf.concat([enc_all[i], tf.tile([[0.0, 1.0, 0.0, 0.0]], [tf.shape(enc_all)[1], 1])], -1)
            elif self.flag[i] == 1:
                enc_new[i] = tf.concat([enc_all[i], tf.tile([[0.0, 0.0, 1.0, 0.0]], [tf.shape(enc_all)[1], 1])], -1)
            else:
                enc_new[i] = tf.concat([enc_all[i], tf.tile([[0.0, 0.0, 0.0, 1.0]], [tf.shape(enc_all)[1], 1])], -1)


        enc_new = tf.convert_to_tensor(enc_new)

        with tf.variable_scope('nn', reuse=tf.AUTO_REUSE):
          enc_en = multihead_attention(queries=enc_new,
                                   keys=enc_new,
                                   values=enc_new,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_en = ff(enc_en, num_units=[4*304, 304])
        

        #encode kl loss
        kl = tf.keras.losses.KLDivergence(reduction = losses_utils.ReductionV2.NONE, name = 'kl')
        kl_loss1 = kl(tf.nn.softmax(enc_en, -1), tf.nn.softmax(self.pretrained_output, -1))
        kl_loss2 = kl(tf.nn.softmax(self.pretrained_output, -1), tf.nn.softmax(enc_en, -1))

        self.kl_loss = tf.reduce_sum(tf.reduce_mean(kl_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(kl_loss2, -1), -1) 


        enc_en = tf.multiply(enc_en, 1, name='encode_outputs')

        #decode 
        with tf.variable_scope('de', reuse=tf.AUTO_REUSE):
          enc_de = multihead_attention(queries=enc_en,
                                   keys=enc_en,
                                   values=enc_en,
                                   num_heads=4,
                                   dropout_rate= 0.2,
                                   training = True,
                                   causality=False)
          enc_de = ff(enc_de, num_units=[4*304, 304])

        de_loss1 = kl(tf.nn.softmax(enc_de, -1), tf.nn.softmax(enc_new, -1))
        de_loss2 = kl(tf.nn.softmax(enc_new, -1), tf.nn.softmax(enc_de, -1))
        self.de_loss = tf.reduce_sum(tf.reduce_mean(de_loss1, -1), -1) + tf.reduce_sum(tf.reduce_mean(de_loss2, -1), -1)
        

        outputs_en = SigmoidAtt(enc_en, Wr_wq, Wm_wq, Wu_wq)

        self.tag_loss = tf.reduce_mean(tf.keras.losses.mae(enc_new[:,1,-4:], tf.nn.sigmoid(outputs_en[:, -4:])))
        temp_new = outputs_en
        temp_new = tf.layers.dense(temp_new, self.config.att_dim, use_bias=False)

        output_res = tf.add(tf.matmul(temp_new, W_l), b_l)
        ouput_label = tf.one_hot(self.label, self.config.class_num)
        self.prob = tf.nn.softmax(output_res)


        with tf.name_scope('loss'):
          loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = output_res, labels=ouput_label))
          self.loss = loss
          self.l2_loss = tf.contrib.layers.apply_regularization(regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                  weights_list = [W_l, b_l])
          self.total_loss = self.loss+ 0.1*self.l2_loss + 0.1*self.kl_loss + 0.1*self.de_loss + 0.1* self.tag_loss






