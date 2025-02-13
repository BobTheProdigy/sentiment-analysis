import torch # pip install torch torchvision torchaudio
import torch.nn as nn
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # defaults to GPU when available, and CPU if not
import time

# Purpose: processes a list of text reviews, creating a numerical representation of the words in the reviews, needed for ML
# Parameters: review (list of strings)
# Return values: vocab, padded, hashing
def lex_order(reviews):
    # makes set of all the words in a review
    startTime = time.time()
    words = set()
    # print(f"Unique words in lex_order_new vocabulary: {len(words)}")
    for i in reviews: # problem: crashed because I traid to make another list of lists of it so I can maintain the order but that was too much for memory 1/3
        for n in i.split(): # solution found: just use the input reviews as the order and don't make a list of lists 1/7
            words.add(n)
    words = sorted(list(words))
    vocab = len(words) + 1
    hashing = {}
    order = 1
    for i in words:
        hashing[i] = order
        order += 1
    lex = []
    for i in reviews:
        lister = []
        for n in i.split():
            lister.append(hashing[n])
        lex.append(torch.tensor(lister)) # tensors can't change size so make list of tensors, DEFAULTS TO CPU
    # torch method below used to add padding to make all tensors (similar to arrays in java) same length
    padded = torch.nn.utils.rnn.pad_sequence(lex, batch_first=True)
    print("lex_order() runtime: " + str(round(time.time() - startTime, 2)))
    return vocab, padded, hashing # solution found: use secound smaller dataset after discussing with proffessor - 1/7
    #return hashing bc I want to test new data