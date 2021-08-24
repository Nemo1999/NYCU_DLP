"""
This is the specification file for conditional seq2seq VAE training and validation dataset - English tense

1. train.txt
The file is for training. There are 1227 training pairs.
Each training pair includes 4 words: simple present(sp), third person(tp), present progressive(pg), simple past(p).


2. test.txt
The file is for validating. There are 10 validating pairs.
Each training pair includes 2 words with different combination of tenses.
You have to follow those tenses to test your model.

Here are to details of the file:

sp -> p
sp -> pg
sp -> tp
sp -> tp
p  -> tp
sp -> pg
p  -> sp
pg -> sp
pg -> p
pg -> tp

"""
from io import open
import torch

def ch2ord(ch):
    assert type(ch) == str , "expected character as input"
    assert len(ch) == 1, "a single character expected, got string"
    ch = ch.lower()
    if ord('a') <= ord(ch) <= ord('z'):
        return ord(ch) - ord('a')
    elif ch == '^': # start of sequence
        return ord('z')+1
    elif ch == '$': # end of sequence
        return ord('z')+2
    else:
        assert False, "unknown character"

def ord2ch(n):
    assert type(n) == int, f"expect integer, got {type(n)}"
    if n <= 25:
        return chr(n + ord('a'))
    elif n == 26: # start of sequence
        return '^'
    elif n == 27: # end of sequence
        return '$'
    else:
        assert False, "unknown code"

total_char = 28

total_tense = 4

def get_training_pairs():
    """
    Return pairs of training word and their tense (both encoded)
    characters of words are encoded using `ord` and `chr`
    tense are encoded using
     sp -> 0
     tp -> 1
     pg -> 2
     p  -> 3
    """
    lines = open('dataset/train.txt').read().strip().split('\n')
    pairs = []
    for l in lines:
        for tense , word in enumerate(l.split(' ')):
            word = [ch2ord(ch) for ch in word]
            word.append(27)
            word = torch.tensor(word,dtype=torch.long).view(-1,1)
            tense = torch.tensor(tense,dtype=torch.long).view(-1,1)
            pairs.append((word, tense))
    return pairs

def get_testing_pairs():
    lines = open('dataset/test.txt').read().strip().split('\n')
    pairs = []

    validation_terms = [
        (0,3),
        (0,2),
        (0,1),
        (0,1),
        (3,1),
        (0,2),
        (3,0),
        (2,0),
<<<<<<< HEAD
        (2,3),
=======
        (2,4),
>>>>>>> ccfb4c8982e1c26908e84288185ac7131ac240aa
        (2,1)
    ]
    for l in lines:
        for (t1,t2) in validation_terms:
            w1, w2 = l.split(' ')
            w1, w2 = list(map(ch2ord,w1)), list(map(ch2ord,w2))
            w1.append(26) # append EOS symbol
            w2.append(26) # append EOS symbol
            w1 = torch.tensor(w1,dtype=torch.long).view(-1,1)
            w2 = torch.tensor(w2,dtype=torch.long).view(-1,1)
            t1 = torch.tensor([t1],dtype=torch.long).view(-1,1)
            t2 = torch.tensor([t2],dtype=torch.long).view(-1,1)
            pairs.append(((w1,t1),(w2,t2)))
    return pairs

def nums2word(nums):
    return ''.join([ord2ch(int(n)) for n in nums])



if __name__ == '__main__':
    print(get_training_pairs()[0])
    print(get_testing_pairs()[0])
    print(nums2word(get_testing_pairs()[0][0][0]))
