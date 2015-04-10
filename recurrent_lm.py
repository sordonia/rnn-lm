"""
General class for a recurrent language model
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Alessandro Sordoni")
__contact__ = "Alessandro Sordoni <sordonia@iro.umontreal>"

import theano
import theano.tensor as T
import numpy as np
import cPickle
import logging
import operator

logger = logging.getLogger(__name__)

from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict

from model import *
from utils import *

# Theano speed-up
theano.config.scan.allow_gc = False
#

def add_to_params(params, new_param):
    params.append(new_param)
    return new_param

class ComponentBase():
    def __init__(self, state, rng, parent):
        patience = state['patience'] 
        self.rng = rng
        self.trng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))
        
        self.parent = parent
        self.state = state
        self.__dict__.update(state)
        
        self.rec_activation = eval(self.rec_activation)
        self.params = []

class LanguageModel(ComponentBase): 
    TRAINING = 0
    EVALUATION = 1
    SAMPLING = 2
    BEAM_SEARCH = 3
    
    def init_params(self):
        ###################
        # RECURRENT WEIGHTS
        ###################
        self.W_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='W_emb'))
        self.W_in = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in'))
        self.W_hh = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh'))
        self.b_hh = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_hh'))
        
        if self.step_type == "gated":
            self.b_r = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_r'))
            self.b_z = add_to_params(self.params, theano.shared(value=np.zeros((self.qdim,), dtype='float32'), name='b_z'))
            
            self.W_in_r = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_r'))
            self.W_in_z = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, self.qdim), name='W_in_z'))
            self.W_hh_r = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_r'))
            self.W_hh_z = add_to_params(self.params, theano.shared(value=OrthogonalInit(self.rng, self.qdim, self.qdim), name='W_hh_z'))
        
        self.bd_out = add_to_params(self.params, theano.shared(value=np.zeros((self.idim,), dtype='float32'), name='bd_out'))
        self.Wd_emb = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.idim, self.rankdim), name='Wd_emb'))
        
        ######################   
        # Output layer weights
        ######################
        out_target_dim = self.qdim
        if not self.maxout_out:
            out_target_dim = self.rankdim

        self.Wd_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.qdim, out_target_dim), name='Wd_out'))
         
        # Set up deep output
        if self.deep_out:
            self.Wd_e_out = add_to_params(self.params, theano.shared(value=NormalInit(self.rng, self.rankdim, out_target_dim), name='Wd_e_out'))
            self.bd_e_out = add_to_params(self.params, theano.shared(value=np.zeros((out_target_dim,), dtype='float32'), name='bd_e_out'))
             
    def plain_step(self, x_t, h_tm1):
        #### Handle the bias from the document
        h_t = T.dot(x_t, self.W_in) + T.dot(h_tm1, self.W_hh) + self.b_hh
        h_t = self.rec_activation(h_t)
        return h_t
     
    def gated_step(self, x_t, h_tm1):
        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_r) + T.dot(h_tm1, self.W_hh_r) + self.b_r)
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_in_z) + T.dot(h_tm1, self.W_hh_z) + self.b_z)
        
        h_tilde = T.dot(x_t, self.W_in) + T.dot(r_t * h_tm1, self.W_hh) + self.b_hh
        h_tilde = self.rec_activation(h_tilde)
        h_t = (np.float32(1.0) - z_t) * h_tm1 + z_t * h_tilde 
        return h_t, r_t, z_t, h_tilde
    
    def approx_embedder(self, x):
        return self.W_emb[x]
     
    def build_lm(self, x, y=None, mode=TRAINING, prev_h=None, step_num=None):
        """
        x is the input sequence
        y are the targets
        mode is the evaluation or sampling mode
        prev_h is used in the sampling mode
        step_num is the step number of decoding
        """ 
        one_step = False
        
        # Check parameter consistency
        if mode == LanguageModel.EVALUATION or mode == LanguageModel.TRAINING:
            assert y
        else:
            assert not y
            assert prev_h
            one_step = True

        # if x.ndim == 2 then 
        # x = (n_steps, batch_size)
        if x.ndim == 2:
            batch_size = x.shape[1]
        # else x = (word_1, word_2, word_3, ...)
        # or x = (last_word_1, last_word_2, last_word_3, ..)
        # in this case batch_size is 
        else:
            batch_size = 1
         
        if not prev_h:
            prev_h = T.alloc(np.float32(0.), batch_size, self.qdim)
         
        xe = self.approx_embedder(x)
        # Gated Encoder
        if self.step_type == "gated":
            f_enc = self.gated_step
            o_enc_info = [prev_h, None, None, None]
        else:
            f_enc = self.plain_step
            o_enc_info = [prev_h]
        
        # Run through all the sentence (encode everything)
        if not one_step: 
            _res, _ = theano.scan(f_enc,
                              sequences=[xe],\
                              outputs_info=o_enc_info) 
        # Make just one step further
        else:
            _res = f_enc(xe, prev_h)

        h = _res[0]
        # Store last h for further use
        pre_activ = self.output_layer(h, xe)
         
        # EVALUATION  : Return target_probs
        # target_probs.ndim == 3
        outputs = self.output_softmax(pre_activ)
            
        if mode == LanguageModel.EVALUATION:
            target_probs = GrabProbs(outputs, y)
            return target_probs, h
        # BEAM_SEARCH : Return output (the softmax layer) + the new hidden states
        elif mode == LanguageModel.BEAM_SEARCH:
            return outputs, h
        # SAMPLING    : Return a vector of n_sample from the output layer 
        #                 + log probabilities + the new hidden states
        elif mode == LanguageModel.SAMPLING:
            if outputs.ndim == 1:
                outputs = outputs.dimshuffle('x', 0)
        
            sample = self.trng.multinomial(pvals=outputs, dtype='int64').argmax(axis=-1)
            if outputs.ndim == 1:
                sample = sample[0]
         
            log_prob = -T.log(T.diag(outputs.T[sample]))
            return sample, log_prob, h
 
    def output_layer(self, h, x):
        pre_activ = T.dot(h, self.Wd_out)

        if self.deep_out:
            pre_activ += T.dot(x, self.Wd_e_out) + self.bd_e_out
             
        if self.maxout_out:
            pre_activ = Maxout(2)(pre_activ)
         
        return pre_activ
    
    def output_softmax(self, pre_activ):
        # returns a (timestep, bs, idim) matrix (huge)
        return SoftMax(T.dot(pre_activ, self.Wd_emb.T) + self.bd_out)

    def build_next_probs_predictor(self, x, prev_h, d=None):
        return self.build_lm(x, d, mode=LanguageModel.BEAM_SEARCH, prev_h=prev_h)
    
    def sampling_step(self, *args): 
        args = iter(args)

        # Arguments that correspond to scan's "sequences" parameteter:
        step_num = next(args)
        assert step_num.ndim == 0
        
        # Arguments that correspond to scan's "outputs" parameteter:
        prev_word = next(args)
        assert prev_word.ndim == 1
        
        # skip the previous word log probability
        log_prob = next(args)
        assert log_prob.ndim == 1
        
        prev_h = next(args) 
        assert prev_h.ndim == 2
        
        # When we sample we shall recompute the lm for one step...
        sample, log_prob, h = self.build_lm(prev_word, prev_h=prev_h, step_num=step_num, mode=LanguageModel.SAMPLING)
        
        assert sample.ndim == 1
        assert log_prob.ndim == 1
        assert h.ndim == 2

        return [sample, log_prob, h]
    
    def build_sampler(self, n_samples, n_steps):
        # For the naive sampler, the states are:
        # 1) a vector [<q>] * n_samples to seed the sampling
        # 2) a vector of [ 0. ] * n_samples for the log_probs
        # 3) prev_h hidden layers
        # TODO: This does not support the document bias
        states = [T.alloc(np.int64(self.sos_sym), n_samples),
                  T.alloc(np.float32(0.), n_samples),
                  T.alloc(np.float32(0.), n_samples, self.qdim)]
        outputs, updates = theano.scan(self.sampling_step,
                    outputs_info=states,
                    sequences=[T.arange(n_steps, dtype='int64')], 
                    n_steps=n_steps,
                    name="sampler_scan")
        # Return sample, log_probs and updates (for tnrg multinomial)
        return (outputs[0], outputs[1]), updates
    ####

    def __init__(self, state, rng, parent):
        ComponentBase.__init__(self, state, rng, parent)
        self.init_params()

class RecurrentLM(Model):
    def indices_to_words(self, seq):
        sen = []
        for k in range(len(seq)):
            sen.append(self.idx_to_str[seq[k]])
            if seq[k] == self.eos_sym:
                break
        return ' '.join(sen)

    def words_to_indices(self, seq):
        sen = []
        for k in range(len(seq)):
            sen.append(self.str_to_idx.get(seq[k], self.unk_sym))
        return sen

    def compute_updates(self, training_cost, params):
        updates = {}
        
        grads = T.grad(training_cost, params)
        grads = OrderedDict(zip(params, grads))
        
        # Clip stuff
        c = numpy.float32(self.cutoff)
        clip_grads = []
        
        norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
        normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
        notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
         
        for p, g in grads.items():
            clip_grads.append((p, T.switch(notfinite, numpy.float32(.1) * p, g * normalization)))
        grads = OrderedDict(clip_grads)

        if self.updater == 'adagrad':
            updates = Adagrad(grads, self.lr)  
        elif self.updater == 'sgd':
            raise Exception("Sgd not implemented!")
        elif self.updater == 'adadelta':
            updates = Adadelta(grads)
        elif self.updater == 'rmsprop':
            updates = RMSProp(grads, self.lr)
        elif self.updater == 'adam':
            updates = Adam(grads)
        else:
            raise Exception("Updater not understood!") 
        return updates
   
    def build_train_function(self):
        if not hasattr(self, 'train_fn'):
            # Compile functions
            logger.debug("Building train function")
            model_updates = self.compute_updates(self.prediction_cost / self.x_data.shape[1], self.params)
            self.train_fn = theano.function(inputs=[self.x_data, self.x_max_length, self.x_cost_mask],
                                            outputs=self.prediction_cost, updates=model_updates, name="train_fn") 
        return self.train_fn

    def build_eval_function(self):
        if not hasattr(self, 'eval_fn'):
            # Compile functions
            logger.debug("Building evaluation function")
            self.eval_fn = theano.function(inputs=[self.x_data, self.x_max_length, self.x_cost_mask], 
                                            outputs=self.prediction_cost, name="eval_fn")
        return self.eval_fn

    def build_sampling_function(self):
        if not hasattr(self, 'sample_fn'):
            logger.debug("Building sampling function")
            self.sample_fn = theano.function(inputs=[self.n_samples, self.n_steps], outputs=[self.sample, self.sample_log_prob], \
                                       updates=self.sampling_updates, name="sample_fn")
        return self.sample_fn

    def build_next_probs_function(self):
        if not hasattr(self, 'next_probs_fn'):
            outputs, h = self.language_model.build_next_probs_predictor(self.beam_source, prev_h=self.beam_h)
            self.next_probs_fn = theano.function(inputs=[self.beam_h, self.beam_source],
                                                 outputs=[outputs, h],
                                                 name="next_probs_fn")
        return self.next_probs_fn
    
    def __init__(self, rng, state):
        Model.__init__(self)    
        self.state = state
         
        # Compatibility towards older models
        self.__dict__.update(state)
        self.rng = rng 

        # Load dictionary 
        raw_dict = cPickle.load(open(self.dictionary, 'r'))
        
        # Probabilities for each term in the corpus
        self.str_to_idx = dict([(tok, tok_id) for tok, tok_id, _ in raw_dict])
        self.idx_to_str = dict([(tok_id, tok) for tok, tok_id, freq in raw_dict])

        if '<s>' not in self.str_to_idx \
           or '</s>' not in self.str_to_idx:
                raise Exception("Error, malformed dictionary!")
         
        # Number of words in the dictionary 
        self.idim = len(self.str_to_idx)
        self.state['idim'] = self.idim
 
        logger.debug("Initializing language model")
        self.language_model = LanguageModel(self.state, self.rng, self)
        
        # Init params
        self.params = self.language_model.params 
        
        self.x_data = T.imatrix('x_data')
        self.x_reset = T.vector('x_reset')
        self.x_cost_mask = T.matrix('cost_mask')
        self.x_max_length = T.iscalar('x_max_length')
        
        # The training is done with a trick. We append a special </q> at the beginning of the session
        # so that we can predict also the first query in the session starting from the session beginning token (</q>).
        self.aug_x_data = T.concatenate([T.alloc(np.int32(self.eos_sym), 1, self.x_data.shape[1]), self.x_data])
        self.training_x = self.aug_x_data[:self.x_max_length]
        self.training_y = self.aug_x_data[1:self.x_max_length+1]
        self.training_x_cost_mask = self.x_cost_mask[:self.x_max_length].flatten()
         
        target_probs, self.eval_h = self.language_model.build_lm(self.training_x, 
                                                    y=self.training_y, 
                                                    mode=LanguageModel.EVALUATION)

        # Prediction cost
        self.prediction_cost = T.sum(-T.log(target_probs) * self.training_x_cost_mask) 
        
        # Sampling variables
        self.n_samples = T.iscalar("n_samples")
        self.n_steps = T.iscalar("n_steps")
        (self.sample, self.sample_log_prob), self.sampling_updates \
                = self.language_model.build_sampler(self.n_samples, self.n_steps) 

        # Beam-search variables
        self.beam_source = T.lvector("beam_source")
        self.beam_h = T.matrix("beam_h")
        self.beam_step_num = T.lscalar("beam_step_num")
