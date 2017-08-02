#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:   Travis A. Ebesu
@created:  2016-11-14
@summary:
'''
from __future__ import unicode_literals
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str)
parser.add_argument('-r', '--restore', help='optional, only if restoring', type=str,
                    default=None)
parser.add_argument('-i', '--input', help='Data input filename', type=str)

args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import sys
import threading
import cPickle
from logging.config import dictConfig
import logging
import json


import tensorflow as tf
import numpy as np

from rbase.citerec import PAD_TOKEN, GO_TOKEN, END_TOKEN, CitationContextDataset
import rbase.seq2seq import DeepCNNtoRNN

DOC_COUNT = 4258383

MODEL = DeepCNNtoRNN


filename = args.input
restore = False

if args.restore:
    restore = True
    restore_directory = args.restore


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

    epoches = 5

    batch_size = 64

    # Embedding Size
    embed_size = 64

    # Cell Unit size
    cell_size = 64

    filter_sizes = [4, 4, 5]

    num_filters = 64

    author_filter_size = [1, 2]

    author_num_filters = 64

    author_embed_size = 64

    use_authors = True

    max_author_len = 5

    # Number of layers
    num_layers = 1

    # Directory to save
    save_directory = None

    # RNN cell type
    cell_type = tf.nn.rnn_cell.GRUCell
    #cell_type = tf.nn.rnn_cell.LSTMCell

    # Arguments to pass to cell type
    cell_args = dict(num_units=cell_size)

    # Feed Previous inputs, ie decoding
    feed_previous = False

    # Initial Learning Rate
    learning_rate = 0.1

    # Optimizer
    optimizer = 'ADAM'
    #tf.train.GradientDescentOptimizer
    # optimizer_args = dict(learning_rate=learning_rate)

    # Dropout if supported
    dropout = 0.8

    # Gradient clipping
    grad_clip = 5.0

    # Bucketing
    buckets = [
        (32, 10),
        (32, 17)]


    # init this
    encoder_vocab_size = None
    decoder_vocab_size = None

    PAD_ID = None
    GO_ID = None

    model = MODEL
    notes = """Adding author information to the attention mechansim"""


    save_directory = None
    _IGNORE = ['fields', 'save', 'load']

    # Set Custom Parameters by name with init
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def fields(self):
        """
        Get all fields/properties stored in this config class
        """
        return [m for m in dir(self)
                if not m.startswith('_') and m not in self._IGNORE]

    def save(self):
        """
        Config is dumped as a json file
        """
        json.dump(self._get_dict(),
                  open('%s/config.json' % self.save_directory, 'w'),
                  sort_keys=True, indent=2)
        pickle.dump({key: self.__getattribute__(key) for key in self.fields},
                    open('%s/config.pkl' % self.save_directory, 'wb'),
                    pickle.HIGHEST_PROTOCOL)

    def load(self):
        """
        Load config, equivalent to loading json and updating this classes' dict
        """
        try:
            d = pickle.load(open('%s/config.pkl' % self.save_directory))
            self.__dict__.update(d)
        except Exception:
            d = json.load(open('%s/config.json' % self.save_directory))
            self.__dict__.update(d)

    def _get_dict(self):
        return {key: self.__getattribute__(key) if isinstance(self.__getattribute__(key), (int, long, float))
                else unicode(self.__getattribute__(key))  for key in self.fields}

    def __repr__(self):
        return json.dumps(self._get_dict(), sort_keys=True, indent=2)

    def __str__(self):
        return json.dumps(self._get_dict(), sort_keys=True, indent=2)


class BucketingQueue(object):

    def __init__(self, config, reverse=True, capacity=2048, threads=1):
        batch_size, bucket_count = config.batch_size, len(config.buckets)
        with tf.variable_scope('BucketingQueue'):
            # bucket_id, encoder_inputs, decoder_inputs, target_weights
            self.queue = tf.PaddingFIFOQueue(capacity=capacity,
                                             dtypes=[tf.int32, tf.int32, tf.int32, tf.int32,
                                                     tf.float32, tf.int32, tf.int32, tf.int32],
                                             shapes=[(), (None, ), (None, ), (None, ),
                                                     (None,), (None,), (None,), (None,)],
                                            name="BucketingFIFOQueue")

            self.bucket_id = tf.placeholder(tf.int32, [], name='BucketId')
            self.encoder_inputs = tf.placeholder(tf.int32, [None, ], name='EncoderInputs')
            self.decoder_inputs = tf.placeholder(tf.int32, [None, ], name='DecoderInputs')
            self.weights = tf.placeholder(tf.int32, [None, ], 'DecoderWeights')
            self.targets = tf.placeholder(tf.float32, [None,], 'DecoderTargets')
            self.decoder_length = tf.placeholder(tf.int32, [None, ], name='DecoderLength')

            self.encoder_authors = tf.placeholder(tf.int32, [None, ], name='AuthorEncoderInputs')
            self.decoder_authors = tf.placeholder(tf.int32, [None, ], name='AuthorDecoderInputs')

            self._batch_size = batch_size


            self._enqueue_op = self.queue.enqueue([self.bucket_id, self.encoder_inputs,
                                                   self.decoder_inputs,
                                                   self.weights, self.targets,
                                                   self.decoder_length, self.encoder_authors,
                                                   self.decoder_authors])

            bucket_id, encoder, decoder, weights, targets, dec_len, enc_authors, dec_authors = self.queue.dequeue()

            self.bucket = tf.contrib.training.bucket([encoder, decoder, weights, targets, dec_len,
                                                      enc_authors, dec_authors],
                                                     bucket_id, batch_size, bucket_count,
                                                     capacity=capacity,
                                                     num_threads=threads,
                                                     dynamic_pad=True, name="BucketingOp")
            self._close_op = self.queue.close()
        self._MAX_AUTHOR = config.max_author_len
        self._config = config
        self._buckets = config.buckets
        self.PAD_ID = config.PAD_ID
        self.GO_ID = config.GO_ID
        self._reverse = reverse
        self.END_ID = config.END_ID

    def _get_batch(self, encoder_seq, decoder_seq):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        bucket_id = self._get_bucket(encoder_seq, decoder_seq)
        encoder_size, decoder_size = self._buckets[bucket_id]
        # Padding Amount
        enc_pad = encoder_size - len(encoder_seq)
        dec_pad = decoder_size - len(decoder_seq) - 1

        weights = np.ones(decoder_size, dtype=np.float32)
        decoder_input = np.array([self.GO_ID] + decoder_seq + [self.PAD_ID] * dec_pad, dtype=np.int32)
        encoder_input = np.array(encoder_seq + [self.PAD_ID] * enc_pad, dtype=np.int32)

        if self._reverse:
            encoder_input = encoder_input[::-1]

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            # We set weight to 0 if the corresponding target is a PAD symbol.
            # The corresponding target is decoder_input shifted by 1 forward.
            if length_idx < decoder_size - 1:
                target = decoder_input[length_idx + 1]
            if length_idx == decoder_size - 1 or target == self.PAD_ID:
                weights[length_idx] = 0.0
        return encoder_input, decoder_input, weights



    def _get_bucket(self, encoder, decoder):
        """
        Given a parallel sequence and buckets, it finds the bucket id
        """
        src_len = len(encoder)
        src_tgt = len(decoder) + 1

        for bucket_id, (source_size, target_size) in enumerate(self._buckets):
            # source_id < bucket size and length(target) < bucket size
            if src_len < source_size and src_tgt < target_size:
                return bucket_id
        raise Exception("Bad Bucketing... Could not Fit Sequence (%s, %s) in buckets %s" % (src_len, src_tgt, self._buckets))

    def add(self, sess, encoder, decoder, enc_author, dec_author):
        bucket_id = self._get_bucket(encoder, decoder)


        pad_enc_auth = np.zeros(self._MAX_AUTHOR)
        pad_dec_auth = np.zeros(self._MAX_AUTHOR)

        pad_enc_auth[:len(enc_author)] = enc_author[:self._MAX_AUTHOR]
        pad_dec_auth[:len(dec_author)] = dec_author[:self._MAX_AUTHOR]

        dec_len = len(decoder)
        encoder, decoder, weights = self._get_batch(encoder, decoder)
        targets = np.hstack([decoder[1:], np.zeros(1, dtype=np.int32)])

        sess.run(self._enqueue_op, {self.bucket_id: bucket_id,
                                    self.encoder_inputs: encoder,
                                    self.weights: weights,
                                    # Shift the decoder to the left one
                                    self.targets: targets,
                                    self.decoder_inputs: decoder,
                                    self.decoder_length: [dec_len],
                                    self.encoder_authors: pad_enc_auth,
                                    self.decoder_authors: pad_dec_auth})



def create_exp_directory(cwd=''):
    '''
    Creates a new directory to store experiment to save data

    Folders: XXX, creates directory sequentially

    Returns
    -------
    exp_dir : str
        The newly created experiment directory

    '''
    created = False
    for i in range(1, 10000):
        exp_dir = str(i).zfill(3)
        path = os.path.join(cwd, exp_dir)
        if not os.path.exists(path):
            # Create directory
            os.mkdir(path)
            created = True
            break
    if not created:
        print 'Could not create directory for experiments'
        exit(-1)
    return path + '/'

def BatchInputSequences(coord):
    train_idx = dataset.getTrainIndex()

    while True:
        np.random.shuffle(train_idx)
        for idx in train_idx:
            if coord.should_stop():
                break
            example = dataset.getContext(idx)
            encoder = example['context']
            decoder = example['title']


            buckets.add(sess, encoder, decoder,
                        example['cluster_authors'], # Encoder Context: where authors of cited paper
                        example['citing_authors'] # Decoder Paper Title Author
            )

        if coord.should_stop():
            break

dataset = CitationContextDataset(filename)
if restore:
    config = Config(save_directory=restore_directory)
    config.load()
else:
    config = Config(save_directory=create_exp_directory('/home/tebesu/result/citerec/'))

# Setup Logging
LOGGING_CONFIG = dict(
    version=1,
    formatters={
        # For files
        'detailed': {
            'format': "[%(asctime)s - %(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s",
        },
        # For the console
        'console': {
            'format':"[%(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s",
        }
    },
    handlers={
        'console': {
            'class': 'logging.StreamHandler',
            'level': logging.DEBUG,
            'formatter': 'console',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': logging.DEBUG,
            'formatter': 'detailed',
            'filename': "{}/log".format(config.save_directory),
            'mode': 'a',
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 5
        }
    },
    root={
        'handlers': ['console', 'file'],
        'level': logging.DEBUG,
    },
    disable_existing_loggers=False,
)

dictConfig(LOGGING_CONFIG)

if restore:
    tf.logging.info('Resuming from directory: %s' % restore_directory)

df = cPickle.load(open('title_context_df.pkl'))


train_idx = dataset.getTrainIndex()
encoder_vocab = dataset.getEncoderVocab()
decoder_vocab = dataset.getDecoderVocab()

author_vocab = dataset.getPickle("author", 'data')

sv_config = {
    'logdir': config.save_directory,
    'checkpoint_basename': 'seq2seq.chkpt',

    'save_model_secs': 60*10, # 10 minutes

    # No summary
    'summary_op': None,
}

config.encoder_vocab_size=len(encoder_vocab)
config.decoder_vocab_size=len(decoder_vocab)
config.PAD_ID = encoder_vocab.get(PAD_TOKEN)
config.GO_ID = encoder_vocab.get(GO_TOKEN)
config.END_ID = encoder_vocab.get(END_TOKEN)
config.author_vocab_size = len(author_vocab)
epoches = config.epoches

tf.logging.info("Config: %s" % config)

tf.logging.info('Save Directory: %s' % config.save_directory)
tf.reset_default_graph()




models = []
with tf.variable_scope('Model') as vs:
    for i, (enc_len, dec_len) in enumerate(config.buckets):
        if i > 0:
            vs.reuse_variables()
        tf.logging.info('Creating Graph {}/{} with [{}, {}] buckets'.format(i+1, len(config.buckets), enc_len, dec_len))


        models.append(MODEL(config, enc_len, dec_len))


buckets = BucketingQueue(config, threads=2,
                         capacity=2048)

config.save()

tf.logging.info('Initializing Session...')

# assign_ops = [models[0]._enc_embeddings.assign(dataset.getArray('enc_glove6b_50d')),
#               models[0]._dec_embeddings.assign(dataset.getArray('dec_glove6b_50d'))
#           ]

sv = tf.train.Supervisor(**sv_config)

sess = sv.prepare_or_wait_for_session(max_wait_secs=60*5)


# print "Initializing Pretrained Embeddings"
# sess.run(assign_ops)


threads = [
    threading.Thread(target=BatchInputSequences, args=(sv.coord,), name="BatchInputThread"),
]

tf.logging.info('Starting I/O threads...')
for t in threads:
    t.setDaemon(True)
    t.start()



epoch_loss = []
total = len(train_idx)
batch_count = total / config.batch_size


try:
    for epoch in range(epoches):
        if sv.should_stop():
            break
        print
        tf.logging.info('Starting Epoch {}'.format(epoch+1))
        batch_loss = []

        if epoch > 0:
            decay_rate = 0.5
            tf.logging.info('Decaying Learning Rate by {}'.format(decay_rate))
            models[0].decay_learning_rate(sess, decay_rate)

        for _i in xrange(batch_count):
            # Never got the chance to integrate it
            bucket_id, (encoder_inputs, decoder_inputs, weights, targets, decoder_len, enc_auth, dec_auth) = sess.run(buckets.bucket)


            model = models[bucket_id]
            feed = {
                model.encoder_inputs: encoder_inputs,
                model.decoder_inputs: decoder_inputs,
                # Shifted inputs
                model.decoder_targets: targets,
                model.decoder_seqlen: decoder_len.ravel(),
                model.dropout_keep_prob: config.dropout,
                model.encoder_authors: enc_auth,
                model.decoder_authors: dec_auth
            }

            loss, _ = sess.run([model.loss, model.train], feed)

            if _i % 100 == 0 and bucket_id == 0:
                # Not sure, why I can't just call model, throws an error
                # Calling models[1] also throws an error

                if _i % 500 == 0:
                    tf.logging.debug("Iteration: {:,} / {:<10,} Current Batch Loss: {:<10.4f}".format(_i+1, batch_count,
                                                                                                                           loss))

                sv.summary_computed(sess, sess.run(models[0].summary,
                                                   feed_dict=feed))



            batch_loss.append(loss)

        print
        tf.logging.info('Epoch Loss: {:.2f}'.format(np.mean(batch_loss)))
        epoch_loss.append(np.mean(batch_loss))
        if epoch != (epoches-1):
            # Easier than making another saver, simply rename it and supervisor will save a new copy
            os.rename(os.path.join(config.save_directory, 'seq2seq.chkpt'),
                  os.path.join(config.save_directory, 'seq2seq_epoch%s.chkpt' % epoch))

except KeyboardInterrupt:
    print 'Keyboard Interrupt'

print 'Done...'
sv.coord.request_stop()
