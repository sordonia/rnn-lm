from collections import OrderedDict

def prototype_state():
    state = {} 
     
    # Random seed
    state['seed'] = 1234
    # Logging level
    state['level'] = 'DEBUG'

    # These are unknown word placeholders
    state['oov'] = '<unk>'
    # Watch out for these
    state['unk_sym'] = 0
    state['eos_sym'] = 2
    state['sos_sym'] = 1

    state['n_samples'] = 40
    
    # These are end-of-sequence marks
    state['start_sym_sent'] = '<s>'
    state['end_sym_sent'] = '</s>'
    
    # Low-rank approximation activation function
    state['rank_n_activ'] = 'lambda x: x'

    # ----- SIZES ----
    # Dimensionality of hidden layers
    state['qdim'] = 512
    # Dimensionality of low-rank approximation
    state['rankdim'] = 256

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 5
    state['cost_threshold'] = 1.003
    
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adam'
     
    # Batch size
    state['bs'] = 128 
     
    # We take this many minibatches, merge them,
    # sort the sentences according to their length and create
    # this many new batches with less padding.
    state['sort_k_batches'] = 20
    
    # Maximum sequence length / trim batches
    state['seqlen'] = 50

    # Should we use a deep output layer
    # and maxout on the outputs?
    state['deep_out'] = True
    state['maxout_out'] = True
    
    state['step_type'] = 'gated'
    state['rec_activation'] = "lambda x: T.tanh(x)"

    # Maximum number of iterations
    state['max_iters'] = 10
    state['save_dir'] = './'
    
    # ----- TRAINING PROCESS -----
    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 10
    # Validation frequency
    state['validFreq'] = 5000
    # Number of batches to process
    state['loopIters'] = 3000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1
    return state

def prototype_test():
    state = prototype_state()

    state['train_sentences'] = "tests/data/test.word.train.pkl"
    state['valid_sentences'] = "tests/data/test.word.valid.pkl"
    state['dictionary'] = "tests/data/test.dict.pkl" 
    state['save_dir'] = "tests/models/"

    state['prefix'] = "test_"
    
    state['deep_out'] = True 
    state['maxout_out'] = False 

    #
    state['qdim'] = 5
    # Dimensionality of low-rank approximation
    state['rankdim'] = 5
    # 

    state['bs'] = 10
    state['seqlen'] = 50
    return state
