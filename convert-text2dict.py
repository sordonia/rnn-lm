import collections

import numpy
import operator
import os
import sys
import logging
import cPickle

from collections import Counter

def safe_pickle(obj, filename):
    if os.path.isfile(filename):
        logger.info("Overwriting %s." % filename)
    else:
        logger.info("Saving to %s." % filename)
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('text2dict')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="Tab separated session file")
parser.add_argument("--cutoff", type=int, default=-1, help="Vocabulary cutoff (optional)")
parser.add_argument("--dict", type=str, default="", help="External dictionary (optional)")
parser.add_argument("output", type=str, help="Pickle binarized session corpus")
args = parser.parse_args()
freqs = collections.defaultdict(lambda: 1)

###############################
# Part I: Create the dictionary
###############################
if args.dict != "":
    # Load external dictionary
    assert os.path.isfile(args.dict)
    vocab = cPickle.load(open(args.dict, "r"))
    vocab = dict([(x[0], x[1]) for x in vocab])

    # Check consistency
    assert '<unk>' in vocab
    assert '<s>' in vocab
    assert '</s>' in vocab
else:
    word_counter = Counter()

    for count, line in enumerate(open(args.input, 'r')):
        s = [x for x in line.strip().split()]
        word_counter.update(s)

    total_freq = sum(word_counter.values())
    
    logger.info("Read %d sentences " % (count + 1))
    logger.info("Total word frequency in dictionary %d " % total_freq) 

    if args.cutoff != -1:
        logger.info("Cutoff %d" % args.cutoff)
        vocab_count = word_counter.most_common(args.cutoff)
    else:
        vocab_count = word_counter.most_common()

    vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = len(vocab)

logger.info("Vocab size %d" % len(vocab))

# Some statistics 
mean_sl = 0.
unknowns = 0.
num_terms = 0.
binarized_corpus = []

for line, document in enumerate(open(args.input, 'r')):
    binarized_document = []
    
    for word in document.strip().split():
        word_id = vocab.get(word, 0)
        if not word_id:
            unknowns += 1

        binarized_document.append(word_id)
        freqs[word_id] += 1
     
    binarized_document = [1] + binarized_document + [2]
    freqs[1] += 1
    freqs[2] += 1
    
    document_len = len(binarized_document)
    num_terms += document_len
    
    binarized_corpus.append(binarized_document)

logger.info("Vocab size %d" % len(vocab))
logger.info("Number of unknowns %d" % unknowns)
logger.info("Number of terms %d" % num_terms)
logger.info("Writing training %d documents " % len(binarized_corpus)) 

safe_pickle(binarized_corpus, args.output + ".word.pkl")

# Store triples word, word_id, freq
if args.dict == "":
    safe_pickle([(word, word_id, freqs[word_id]) for word, word_id in vocab.items()], args.output + ".dict.pkl")
