import tensorflow as tf
import os
import sys
import datetime
import numpy as np
from Settings import Config
from Dataset import Dataset
from network import MM
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import logging
logging.getLogger('tensorflow').disabled = True

os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_boolean('train', True, 'set True to train')

traindata = pickle.load(open('../data/iemocap/train.pkl', 'rb'))
testdata  = pickle.load(open('../data/iemocap/test.pkl', 'rb'))

def evaluation(y_pred, y_true):
    f1_s = f1_score(y_true, y_pred, average='macro')
    accuracy_s = accuracy_score(y_true, y_pred)
    return f1_s, accuracy_s

def is_equal(a, b):
    flag = 1
    for i in range(len(a)):
        if a[i] != b[i]:
            flag = 0
            break
    return flag

def train(sess, setting):
    with sess.as_default():
         dataset = Dataset()
         initializer = tf.contrib.layers.xavier_initializer()
         with tf.variable_scope('model', reuse = None, initializer = initializer):
             m = MM(is_training=FLAGS.train)

         optimizer = tf.train.AdamOptimizer(setting.learning_rate)
         global_step = tf.Variable(0, name='global_step', trainable = False)
         train_op = optimizer.minimize(m.total_loss, global_step=global_step)
         sess.run(tf.initialize_all_variables())
         saver = tf.train.Saver(max_to_keep=None)


         new_saver = tf.train.import_meta_graph('./saved-model/MT_ATT_model-290.meta')
         new_saver.restore(sess, './saved-model/MT_ATT_model-290')
         graph = tf.get_default_graph()

         visual = sess.graph.get_tensor_by_name("model/visual:0")
         text = sess.graph.get_tensor_by_name("model/text:0")
         audio = sess.graph.get_tensor_by_name("model/audio:0")
         label = sess.graph.get_tensor_by_name("model/label:0")
         flag = sess.graph.get_tensor_by_name("model/flag:0")

         for epoch in range(setting.epoch_num):
            for i in range(int(len(traindata['L'])/setting.batch_size)):
                cur_batch = dataset.nextBatch(traindata, testdata, FLAGS.train)
                feed_dict = {visual: cur_batch['V'], text: cur_batch['T'], audio: cur_batch['A'], label: cur_batch['L'], flag: cur_batch['F']}
                enc = sess.graph.get_tensor_by_name("model/encode_outputs:0")
                enc_out = sess.run(enc, feed_dict)
                feed_dict = {}
                feed_dict[m.visual] = cur_batch['V']
                feed_dict[m.audio] = cur_batch['A']
                feed_dict[m.text] = cur_batch['T']
                feed_dict[m.label] = cur_batch['L']
                feed_dict[m.flag] = cur_batch['F']
                feed_dict[m.pretrained_output] = enc_out
             
                temp, step, loss_, kl_loss = sess.run([train_op, global_step, m.total_loss, m.kl_loss], feed_dict)
                if step>=50 and step%10==0:
                    time_str = datetime.datetime.now().isoformat()
                    tempstr = "{}: step {}, softmax_loss {:g}, kl_loss {:g}".format(time_str, step, loss_, kl_loss)
                    print (tempstr)
                    path = saver.save(sess, './model/MT_ATT_model', global_step=step)



def test(sess, setting):
    dataset = Dataset()
    with sess.as_default():
        with tf.variable_scope('model'):
            mtest = MM(is_training=FLAGS.train)
        saver = tf.train.Saver()
        testlist = range(100, 10000, 10)
        best_model_iter = -1
        best_model_f1 = -1
        best_model_acc = -1

        for model_iter in testlist:
            try:
                saver.restore(sess, './model/MT_ATT_model-'+str(model_iter))
            except Exception:
                continue
            total_pred = []
            total_y = []
            for i in range(int(len(testdata['L'])/setting.batch_size)):
                cur_batch = dataset.nextBatch(traindata, testdata, FLAGS.train)
                feed_dict = {}
                feed_dict[mtest.visual] = cur_batch['V']
                feed_dict[mtest.audio] = cur_batch['A']
                feed_dict[mtest.text] = cur_batch['T']
                feed_dict[mtest.label] = cur_batch['L']
                feed_dict[mtest.flag] = cur_batch['F']
                feed_dict[mtest.pretrained_output] = list(np.zeros((32,300,304)))
                prob = sess.run([mtest.prob], feed_dict)
                
                for j in range(len(prob[0])):
                    total_pred.append(np.argmax(prob[0][j], -1))
                for item in cur_batch['L']:
                    total_y.append(item)

            f1,accuracy=evaluation(total_pred,total_y)
            if f1>best_model_f1:
                best_model_f1=f1
                best_model_iter=model_iter

            if accuracy>best_model_acc:
                best_model_acc = accuracy

            print ('model_iter:',model_iter)
            print ('f1 score:',f1)
            print ('accuracy score:',accuracy)


        print ('----------------------------')
        print ('best model_iter', best_model_iter)
        print ('best f1 score: ', best_model_f1)
        print ('best accuracy score:', best_model_acc)
        








def main(_):
#    print (FLAGS.train)
#    sys.exit()
    setting = Config()
    with tf.Graph().as_default():
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        if FLAGS.train == True:
            train(sess, setting)
        else:
            test(sess, setting)

if __name__ == '__main__':
    tf.app.run()








