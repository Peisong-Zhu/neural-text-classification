import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import utils
import numpy as np
import random
# settings
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 64, 'Batch_size')
flags.DEFINE_integer('epochs', 1000, 'epochs')
flags.DEFINE_integer('classes', 5, 'class num')
flags.DEFINE_integer('hidden_size', 50, 'number of hidden units')
flags.DEFINE_string('checkpoint_maxacc', 'data/checkpoint_maxacc/', 'checkpoint dir')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('max_grad_norm', 5.0, 'max-grad-norm')
flags.DEFINE_string('embedding_file', 'data/word_embedding.txt', 'embedding_file')
flags.DEFINE_string('traindata', 'data/train_', 'traindata')
flags.DEFINE_string('testdata', 'data/test_', 'testdata')
flags.DEFINE_string('devdata', 'data/dev_', 'devdata')
flags.DEFINE_string('device', '/cpu:0', 'device')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

tflog_dir = os.path.join('data/', 'tflog')

word_embedding = utils.load_embedding(FLAGS.embedding_file)

def HAN_model_1(session, restore_only=False):
  """Hierarhical Attention Network"""
  from models import HAN
  is_training = tf.placeholder(dtype=tf.bool, name='is_training')
  cell = tf.contrib.rnn.GRUCell
  model = HAN(
      #vocab_size=vocab_size,
      word_cell=cell,
      sentence_cell=cell,
      word_output_size=100,
      sentence_output_size=100,
      embedding_size=200,
      #embedding_matrix=tf.stack(word_embedding),
      classes=FLAGS.classes,
      max_grad_norm=FLAGS.max_grad_norm,
      hidden_size=FLAGS.hidden_size,
      learning_rate=FLAGS.lr,
      is_training=is_training,
      device=FLAGS.device,
      dropout_keep_proba=0.5,
      # max_grad_norm=args.max_grad_norm,
  )
  return model

def AN_model_1(session, restore_only=False):
  """Hierarhical Attention Network"""
  from models import AN
  is_training = tf.placeholder(dtype=tf.bool, name='is_training')
  cell = tf.contrib.rnn.GRUCell
  model = AN(
      #vocab_size=vocab_size,
      word_cell=cell,
      word_output_size=100,
      embedding_size=256,
      #embedding_matrix=tf.stack(word_embedding),
      classes=FLAGS.classes,
      max_grad_norm=FLAGS.max_grad_norm,
      hidden_size=FLAGS.hidden_size,
      learning_rate=FLAGS.lr,
      is_training=is_training,
      device=FLAGS.device,
      dropout_keep_proba=0.5,
      # max_grad_norm=args.max_grad_norm,
  )
  return model

model_fn = HAN_model_1
# model_fn = AN_model_1

def batch_iterator(datax, datay, batch_size):
    xb = []
    yb = []
    num = len(datay)/batch_size
    list = range(num)
    random.shuffle(list)
    for i in list:
        #print i
        for j in range(batch_size):
            index = i * batch_size + j
            x = datax[index]
            y = datay[index]
            xb.append(x)
            yb.append(y)
        yield xb, yb
        xb, yb = [], []
    for i in range(num * batch_size, len(datay)):
        x = datax[i]
        y = datay[i]
        xb.append(x)
        yb.append(y)
    yield xb, yb
    xb, yb = [], []

def batch_iteratorx(datax, batch_size):
    xb = []
    num = len(datax)/batch_size
    list = range(num)
    for i in list:
        #print i
        for j in range(batch_size):
            index = i * batch_size + j
            x = datax[index]
            xb.append(x)
        yield xb
        xb = []
    for i in range(num * batch_size, len(datax)):
        x = datax[i]
        xb.append(x)
    yield xb
    xb = []

def evaluate(session, model, datax, datay):
    predictions = []
    labels = []
    lossAll = 0.0
    for x, y in batch_iterator(datax, datay, FLAGS.batch_size):
        labels.extend(y)
        #pred, loss = session.run([model.prediction, model.loss], model.get_feed_data(x, word_embedding, is_training=False))
        pred, loss = session.run([model.prediction, model.loss], model.get_feed_data(x,word_embedding,y,is_training=False))
        predictions.extend(pred)
        lossAll += loss
        if len(labels) == FLAGS.batch_size:
            print labels
            print predictions
    n = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            n += 1
    return float(n)/float(len(labels)), lossAll

def evaluatex(session, model, datax):
    predictions = []
    for x in batch_iteratorx(datax, FLAGS.batch_size):
        pred = session.run([model.prediction, model.loss], model.get_feed_data(x,word_embedding,is_training=False))
        predictions.extend(pred)
    return predictions

def train():
    train_x, train_y = utils.load_data(FLAGS.traindata + 'x.txt', FLAGS.traindata + 'y.txt')
    dev_x, dev_y = utils.load_data(FLAGS.devdata + 'x.txt', FLAGS.devdata + 'y.txt')
    # train_x, train_y = utils.load_data2(FLAGS.traindata + 'x.txt', FLAGS.traindata + 'y.txt')
    # dev_x, dev_y = utils.load_data2(FLAGS.devdata + 'x.txt', FLAGS.devdata + 'y.txt')
    tf.reset_default_graph()
    fres = open('data/res_newattention0.001.txt','a')
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as s:
        model = model_fn(s)
        saver = tf.train.Saver(tf.global_variables())
        s.run(tf.global_variables_initializer())
        # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_maxacc)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(s, ckpt.model_checkpoint_path)
        summary_writer = tf.summary.FileWriter(tflog_dir, graph=tf.get_default_graph())
        cost_val = []
        maxvalacc = 0.0
        valnum = 0
        stepnum = 0
        is_earlystopping = False
        for i in range(FLAGS.epochs):
            print("Epoch:", '%04d' % (i + 1))
            fres.write("Epoch: %02d \n" % (i + 1))
            for j, (x, y) in enumerate(batch_iterator(train_x, train_y,FLAGS.batch_size)):
                t0 = time.time()
                fd = model.get_feed_data(x,word_embedding,y)
                step, summaries, labels, prediction, loss, accuracy, _ = s.run([
                    model.global_step,
                    model.summary_op,
                    model.labels,
                    model.prediction,
                    model.loss,
                    model.accuracy,
                    model.train_op,
                ], fd)
                summary_writer.add_summary(summaries, global_step=step)
                stepnum = stepnum  + 1
                # print labels
                # print prediction
                print("Step:", '%05d' % stepnum, "train_loss=", "{:.5f}".format(loss),
                      "train_acc=", "{:.5f}".format(accuracy),"time=", "{:.5f}".format(time.time() - t0))
                if stepnum % 500 == 0:
                    valnum += 1
                    valacc, valloss = evaluate(s, model, dev_x, dev_y)
                    cost_val.append(valloss)
                    if valacc > maxvalacc:
                        maxvalacc = valacc
                        saver.save(s, FLAGS.checkpoint_maxacc + 'model.ckpt', global_step=step)

                    print("Validation:", '%05d' % valnum,"val_loss=", "{:.5f}".format(valloss),
                          "val_acc=", "{:.5f}".format(valacc), "time=", "{:.5f}".format(time.time() - t0))
                    fres.write("Validation: %05d val_loss= %.5f val_acc= %.5f \n" % (valnum,valloss,valacc))

                    if valnum > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
                        is_earlystopping = True
                        print("Early stopping...")
                        fres.write("Early stopping...\n")
                        break
            if is_earlystopping:
                break
        print("Optimization Finished!")
        fres.write("Optimization Finished!\n")

        #Testing
        test_x, test_y = utils.load_data(FLAGS.testdata + 'x.txt', FLAGS.testdata + 'y.txt')
        test_acc, test_loss = evaluate(s, model, test_x, test_y)
        print("Test set results based last pamameters: cost= %.5f accuracy= %.5f \n"%(test_loss,test_acc))
        fres.write("Test set results based last pamameters: cost= %.5f accuracy= %.5f \n"%(test_loss,test_acc))

        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_maxacc)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(s, ckpt.model_checkpoint_path)
        test_vma_acc, test_vma_loss = evaluate(s, model, test_x, test_y)
        print("Test set results based max-val-acc pama: cost= %.5f accuracy= %.5f \n"%(test_vma_loss,test_vma_acc))
        fres.write("Test set results based max-val-acc pama: cost= %.5f accuracy= %.5f \n" % (test_vma_loss, test_vma_acc))

        # test_x = utils.load_x(FLAGS.testdata + 'x.txt')
        # ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_maxacc)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(s, ckpt.model_checkpoint_path)
        # prediction = evaluatex(s, model, test_x)
        #
        # for i in range(len(prediction)):
        #     fres.write(str(prediction[i]) + '\n')

def main():
    train()

if __name__ == '__main__':
    main()
