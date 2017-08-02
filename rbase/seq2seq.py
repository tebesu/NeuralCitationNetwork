#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:   Travis A. Ebesu
@created:  2016-11-04
@summary:
'''
import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from tensorflow.python.ops.rnn_cell import _linear
from tensorflow.python.ops.seq2seq import _extract_argmax_and_embed

def sequence_mask(max_len, seq_len, batch_size):
    """
    Create a mask that we will use for the cost function

    This mask is the same shape as x and y_, and is equal to 1 for all non-PAD time
    steps (where a prediction is made), and 0 for all PAD time steps (no pred -> no loss)
    The number 30, used when creating the lower_triangle_ones matrix, is the maximum
    sequence length in our dataset

    see: http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
    Creates 1's lower triangular matrix
    eg [[1, 0, 0],
       [1, 1, 0],
       [1, 1, 1]]
    np.tril(np.ones([10, 10]))


    Then gather, according to ith sequence length we pass
    eg [[1, 0, 0],
       [1, 1, 0],
       [1, 1, 1]]
    Takes the length of the sequence row - 1
    if our sequence length is 1, we take the first row
    [1, 0, 0] as our mask

    op = tf.gather(A, [[0], [0], [1]])

    We then slice to the batch size and maximum sequence length for this batch
    tf.slice(tf.gather(lower_triangular_ones, seqlen - 1),\
                           [0, 0], [batch_size, tf.reduce_max(seqlen)])
    """
    lower_triangular_ones = tf.constant(np.tril(np.ones([max_len, max_len])),dtype=tf.float32)
    return tf.slice(tf.gather(lower_triangular_ones, seq_len - 1),\
                           [0, 0], [batch_size, max_len])

class BaseRNN(object):
    """
    Base Seq2Seq Model. Creates the following
    - learning rate and learning rate decay op
    - RNN cells

    Config expects the following properties:
    - learning_rate (float)
    - cell_type (callable, rnn cell type)
    - cell_args (dict, arguments for the rnn cell type)
    - num_layers (int, number of RNN layers)
    """

    def __init__(self, config):
        self.config = config

        with tf.name_scope("LearningRateDecay"):
            self.learning_rate = tf.Variable(float(self.config.learning_rate),
                                             trainable=False, dtype=tf.float32)
            # Placeholder to decay learning rate by some amount
            self._learning_rate_decay_factor = tf.placeholder(tf.float32,
                                                              name='LearningRateDecayFactor')

            # Operation to decay learning rate
            self._learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * self._learning_rate_decay_factor)

        # Add our RNN cells
        self.cell = self.config.cell_type(**self.config.cell_args)

        if self.config.num_layers > 1:
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell
                                                     for _ in range(self.config.num_layers)])



    def decay_learning_rate(self, session, learning_rate_decay):
        """
        Decay the current learning rate by decay amount
        New Learning Rate = Current Learning Rate * Rate Decay
        """
        session.run(self._learning_rate_decay_op,
                    {self._learning_rate_decay_factor: learning_rate_decay})


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False,
                      loop_function=None,
                     feed_previous=False):
    """
    This is the TensorFlow tutorial version

    h_i = rnn(y, c, h_{i-1}) = rnn([Wx + Cc], h_{i-1})
    output = softmax(Wh_i + b)

    This is the tensorflow attention decoder function. Removed the number of heads
    using because we will need to change this function for research purposes.

    RNN decoder with attention for the sequence-to-sequence model.
    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.
    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: rnn_cell.RNNCell defining the cell function and size.
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
          If True, initialize the attentions from the initial state and attention
          states -- useful when we wish to resume decoding from a previously
          stored decoder state and attention states.
    Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x decoder size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
    Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s"
                     % attention_states.get_shape())

    # Set greedy as the default
    if feed_previous and loop_function is None:
        loop_function = _extract_argmax_and_embed

    with tf.variable_scope(scope or "AttentionDecoder", dtype=dtype) as scope:
        dtype = scope.dtype

        # Needed for reshaping.
        batch_size = tf.shape(decoder_inputs[0])[0]

        # Attention States: [batch size, seq length, encoder cell size]
        #                = [batch_size x attn_length x attn_size]
        # attn_length: Encoders Seq Length
        # attn_size: Encoder cell size

        attn_length = attention_states.get_shape()[1].value

        # Shape is dynamic
        if attn_length is None:
            attn_length = tf.shape(attention_states)[1]

        # Encoder Hidden Size
        attn_size = attention_states.get_shape()[2].value

        W_a = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
        v = tf.get_variable("AttnV", [attn_size])

        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = tf.reshape(attention_states,
                            [-1, attn_length, 1, attn_size])

        hidden_features = tf.nn.conv2d(hidden, W_a, [1, 1, 1, 1], "SAME")

        state = initial_state

        def attention(query):
            """
            query is equivalent to the previous hidden state from the decoder
            Put attention masks on hidden using hidden_features and query.
            """
            with tf.variable_scope("Attention"):
                # Compute: Uh_i = previous decoder state
                # attn_size = Encoder Cell Size
                y = tf.nn.rnn_cell._linear(query, attn_size, True)

                y = tf.reshape(y, [-1, 1, 1, attn_size])

                s = tf.reduce_sum(
                          v * tf.tanh(hidden_features + y), [2, 3])


                a = tf.nn.softmax(s)
                tf.add_to_collection("attention", a)

                c = tf.reduce_sum(
                  tf.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
            return tf.reshape(c, [-1, attn_size])

        outputs = []
        prev = None
        batch_attn_size = tf.pack([batch_size, attn_size])
        attns = tf.zeros(batch_attn_size, dtype=dtype) # initially zero
        attns.set_shape([None, attn_size])

        if initial_state_attention:
            attns = attention(initial_state)

        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)

            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]

            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            x = tf.nn.rnn_cell._linear([inp, attns], input_size, True)

            # Run the RNN.
            cell_output, state = cell(x, state)

            # Run the attention mechanism on the next state
            if i == 0 and initial_state_attention:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)

            # Apply final output projections
            # [h_i, c_i]
            with tf.variable_scope("AttnOutputProjection"):
                output = tf.nn.rnn_cell._linear([cell_output, attns],
                                                output_size, True)

            if loop_function is not None:
                prev = output
            outputs.append(output)

    return outputs, state



class DeepCNNtoRNN(BaseRNN):
    """
    CNN to RNN w/Optional Author Embeddings

    config.use_authors
    config.author_vocab_size
    config.author_embed_size
    config.max_author_len
    config.author_filter_size = [2]
    config.author_num_filters = 5

    """
    def __init__(self, config, max_encoder_length, max_decoder_length):
        """

        :param config: Config object
        :param max_encoder_length: int, maximum encoder sequence length
        :param max_decoder_length: int, maximum decoder sequence length
        """
        super(DeepCNNtoRNN, self).__init__(config)
        self._use_authors = self.config.use_authors

        ##################################
        # Placeholders
        ##################################
        self.encoder_inputs = tf.placeholder(tf.int32, [self.config.batch_size,
                                                        max_encoder_length],
                                             'EncoderInputs')
        self.decoder_inputs = tf.placeholder(tf.int32, [self.config.batch_size,
                                                        max_decoder_length],
                                             'DecoderInputs')
        self.decoder_targets = tf.placeholder(tf.int32, [self.config.batch_size,
                                                         max_decoder_length],
                                              'DecoderTargets')

        # We compute the sequence mask or weights
        self.decoder_seqlen = tf.placeholder(tf.int32, [self.config.batch_size],
                                             'DecoderSequenceLength')

        # Set default dropout to None
        self.dropout_keep_prob = tf.placeholder_with_default(1.0, [],
                                                             'DropoutPlaceholder')

        # Combine all the pooled features
        # Feature Size (num_filters) * Filter Count (number of filters we applied)
        num_filters_total = self.config.num_filters * len(self.config.filter_sizes)

        if self._use_authors:
            #assert self.config.author_num_filters == self.config.num_filters, "Currently, we require author num filters == num filters "
            max_author_len = self.config.max_author_len


            self.encoder_authors = tf.placeholder(tf.int32, [self.config.batch_size,
                                                             max_author_len], "EncoderAuthors")
            self.decoder_authors = tf.placeholder(tf.int32, [self.config.batch_size,
                                                             max_author_len], "DecoderAuthors")

            tf.add_to_collection("placeholders", self.encoder_authors)
            tf.add_to_collection("placeholders", self.decoder_authors)

            with tf.variable_scope("Author") as scope:
                # Embedding layer
                self._auth_embeddings = tf.get_variable('Embeddings',
                                                       [self.config.author_vocab_size,
                                                        self.config.author_embed_size],
                                                       initializer=tf.random_normal_initializer(stddev=0.01))

                # [batch size, max authors, author embeding size]
                self._auth_encoder_embed = tf.nn.embedding_lookup(self._auth_embeddings,
                                                          self.encoder_authors)
                # [batch size, max authors, author embeding size]
                self._auth_decoder_embed = tf.nn.embedding_lookup(self._auth_embeddings,
                                                          self.decoder_authors)
                # Expand: [batch size, max authors, embed size, 1]
                self._auth_encoder_inputs = tf.expand_dims(self._auth_encoder_embed, -1)
                self._auth_decoder_inputs = tf.expand_dims(self._auth_decoder_embed, -1)
                author_filter_total = self.config.author_num_filters * len(self.config.author_filter_size)

                # Could try tied weights between authors for conv
                # Outputs: [len(filters), [batch size, 1, 1, num_filters]]
                with tf.variable_scope('AuthorEncoder'):
                    auth_enc_outputs = cnn_layer(self._auth_encoder_inputs,
                                                 self.config.author_filter_size,
                                                 self.config.author_num_filters,
                                                 self.config.author_embed_size)

                    # [batch size, # of author filter_sizes, num_filters]
                    self._auth_enc_outputs = tf.squeeze(tf.concat(2, auth_enc_outputs),
                                                        squeeze_dims=[1])

                    self._auth_enc_outputs = tf.reshape(tf.concat(3, self._auth_enc_outputs),
                                                        [-1, author_filter_total])

                    for i in range(self.config.num_layers-1):
                        with tf.variable_scope("FC-%s" % (i+1)):
                            self._auth_enc_outputs = tf.tanh(_linear(self._auth_enc_outputs,
                                                                     len(self.config.author_filter_size) * self.config.num_filters,
                                                                     True))


                with tf.variable_scope('AuthorDecoder'):
                    auth_dec_outputs = cnn_layer(self._auth_decoder_inputs,
                                                 self.config.author_filter_size,
                                                 self.config.author_num_filters,
                                                 self.config.author_embed_size)

                    # [batch size, # of author filter_sizes, num_filters]
                    self._auth_dec_outputs = tf.squeeze(tf.concat(2, auth_dec_outputs),
                                                        squeeze_dims=[1])

                    self._auth_dec_outputs = tf.reshape(tf.concat(3, self._auth_dec_outputs),
                                                        [-1, author_filter_total])


                    for i in range(self.config.num_layers-1):
                        with tf.variable_scope("FC-%s" % (i+1)):
                            # [batch, filters, author feature maps] * W => [batch, filters, encoder feature maps]
                            self._auth_dec_outputs = tf.tanh(_linear(self._auth_dec_outputs,
                                                                     len(self.config.author_filter_size) * self.config.num_filters,
                                                                     True))


                # [batch size, # author filter_sizes * 2, encoder feature maps]
                self._auth_outputs = tf.reshape(tf.concat(1, [self._auth_enc_outputs, self._auth_dec_outputs]),
                                                [-1, len(self.config.author_filter_size) * 2, self.config.num_filters])


        tf.add_to_collection("placeholders", self.encoder_inputs)
        tf.add_to_collection("placeholders", self.decoder_inputs)
        tf.add_to_collection("placeholders", self.decoder_targets)
        tf.add_to_collection("placeholders", self.decoder_seqlen)
        tf.add_to_collection("placeholders", self.dropout_keep_prob)

        # We create a mask for the decoder sequence inputs
        self._decoder_mask = sequence_mask(max_decoder_length,
                                                     self.decoder_seqlen,
                                                     self.config.batch_size)
        # Add dropout
        self.cell = tf.nn.rnn_cell.DropoutWrapper(self.cell, self.dropout_keep_prob)

        ##################################
        # Encoder
        ##################################
        with tf.variable_scope("Encoder") as scope:

            # Embedding layer
            self._enc_embeddings = tf.get_variable('Embeddings',
                                                   [self.config.encoder_vocab_size,
                                                    self.config.embed_size],
                                                   initializer=tf.random_normal_initializer(stddev=0.01))

            # Inputs: [batch size, seq length, embeding size]
            self._enc_inputs = tf.nn.embedding_lookup(self._enc_embeddings,
                                                      self.encoder_inputs)
            # Reshape: [batch, seq length, embedding size, 1]
            self._enc_inputs_expanded = tf.expand_dims(self._enc_inputs, -1)

            pooled_outputs = cnn_layer(self._enc_inputs_expanded, self.config.filter_sizes,
                      self.config.num_filters, self.config.embed_size)

            self._enc_outputs = tf.squeeze(tf.concat(2, pooled_outputs),
                                                       squeeze_dims=[1])


        # [batch size, num filters total]
        enc_output = tf.reshape(tf.concat(3, self._enc_outputs), [-1, num_filters_total])

        # Apply a nonlinear projection
        for i in range(self.config.num_layers-1):
            with tf.variable_scope("FC-%s" % (i+1)):
                enc_output = tf.tanh(_linear(enc_output, num_filters_total, True))

        # [batch size, # of filter_sizes, num_filters]
        self._enc_outputs = tf.reshape(enc_output,
                                       [-1, len(self.config.filter_sizes), self.config.num_filters])

        if self._use_authors:
            # Concatenate the author layer
            print "Author Outputs: ", self._auth_outputs
            print "Encoder Outputs: ", self._enc_outputs
            self._enc_outputs = tf.concat(1,
                                          [self._enc_outputs, self._auth_outputs])
            print "Concatenated: ", self._enc_outputs

        ##################################
        # Decoder
        ##################################
        with tf.variable_scope("Decoder") as scope:
            self._dec_embeddings = tf.get_variable('Embeddings',
                                                   [self.config.decoder_vocab_size,
                                                    self.config.embed_size])

            self._dec_inputs = tf.nn.embedding_lookup(self._dec_embeddings,
                                                      self.decoder_inputs)

            # Transformed: [Seq Length, [Batch Size, Embeddings]]
            self._dec_inputs = [tf.squeeze(i, squeeze_dims=[1])
                          for i in tf.split(1, max_decoder_length, self._dec_inputs)]

            # Attention States: [batch size, seq length, encoder cell size]
            attn_states = self._enc_outputs

            self._dec_initial_state = self.cell.zero_state(self.config.batch_size, tf.float32)

            loop_function = None
            if self.config.feed_previous:
                loop_function = _extract_argmax_and_embed(self._dec_embeddings, update_embedding=False)

            self._dec_outputs, self._dec_state = attention_decoder(
                self._dec_inputs,
                self._dec_initial_state,
                attn_states, self.cell,
                self.config.decoder_vocab_size,
                loop_function=loop_function,
                feed_previous=self.config.feed_previous)


        with tf.name_scope('Prediction'):
            # Apply: W output mapping
            self.logits = self._dec_outputs
            # Final Output: (seq len, batch size, vocab)
            self.preds = tf.nn.softmax(self.logits,
                                       name="Softmax")
        # Reshape as lists
        dec_targets_reshaped = [tf.squeeze(i, squeeze_dims=[1]) for i in
                                tf.split(1, max_decoder_length, self.decoder_targets, name='SplitTargets')]

        decoder_mask_reshaped = [tf.squeeze(i, squeeze_dims=[1]) for i in
                                     tf.split(1, max_decoder_length, self._decoder_mask , name='SplitMasks')]

        self.score = tf.nn.seq2seq.sequence_loss_by_example(self.logits, dec_targets_reshaped, decoder_mask_reshaped,
                                                   average_across_timesteps=True)

        self.loss = tf.nn.seq2seq.sequence_loss(self.logits, dec_targets_reshaped, decoder_mask_reshaped)

        with tf.name_scope("Optimizer"):
            if self.config.optimizer == 'ADAM':
                optimizer = tf.train.AdamOptimizer()
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            # Obtain vars to train
            tvars = tf.trainable_variables()
            # Clip by norm
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars),
                                              config.grad_clip)
            self.train = optimizer.apply_gradients(zip(grads, tvars), name='Train')

        with tf.name_scope('Summary'):
            tf.scalar_summary('Loss', self.loss),
            tf.scalar_summary('LearningRate', self.learning_rate)
        self.summary = tf.merge_all_summaries()
        tf.add_to_collection("predict", self.preds)
        tf.add_to_collection("score", self.score)
        tf.add_to_collection("loss", self.loss)
        tf.add_to_collection("train", self.train)


def cnn_layer(layer_inputs, filters, num_filters, embed_size, pooling='max'):
    """
    Simple wrapper to perform a Convolution + Pooling Layer

    :param layer_inputs: Shape[batch, seq length, embedding size, 1]
    :param filters: list[int] each designating the size of the filter
    :param num_filters: int, Number of filters features
    :param embed_size: int, Embedding size
    :param pooling: str, optional, default use max pooling, Options: max, avg, None
    :returns: list of tensors [len(filters), [batch size, 1, 1, num_filters]]

    if we add channels
    layer_inputs = [batch, seq_len, embed_size, channels]
    W =            [current_filter, embed_size, channels, num_filters]
    """
    pooled_outputs = []
    inp_shape = layer_inputs.get_shape().as_list()

    seq_length = inp_shape[1]
    channels = inp_shape[3]


    for i, current_filter in enumerate(filters):
        with tf.variable_scope("ConvLayerPool{}-{}".format(current_filter, i)):
            # Convolution Layer
            filter_shape = [current_filter, embed_size, channels, num_filters]
            # normal distribution a standard deviation of `sqrt(3. / (in + out))
            # tf.random_normal_initializer(3.0 / sum(filter_shape))
            W = tf.get_variable('W', filter_shape,
                                initializer=tf.random_normal_initializer(stddev=0.01))

            # If using reLu use a small number so we fire function
            b = tf.get_variable('b', num_filters,
                                initializer=tf.constant_initializer(0.01))
            # Inputs: layer_inputs = [batch, seq length, embedding size, 1 (channels)]
            #         W            = [filter, ebmed size, 1 (channels), num filter/feature map]
            # Outputs: [batch size, sequence_length - filter + 1, 1, num_filters]
            conv = tf.nn.conv2d(
                layer_inputs,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="Conv")

            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="ReLU")

            # Perform Pooling
            # Pooling Output: [batch_size, 1, 1, num_filters].
            if pooling.lower() == 'max':
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, seq_length - current_filter + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="MaxPool")
            elif pooling.lower() == 'avg':
                pooled = tf.nn.avg_pool(
                    h,
                    ksize=[1, seq_length - current_filter + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="AvgPool")
            else:
                pooled = h

            pooled_outputs.append(pooled)

    return pooled_outputs
