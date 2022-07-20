import os
import sys
from tqdm.notebook import tqdm
import numpy as np
from collections import Counter, defaultdict

from nltk.tokenize import sent_tokenize

import roberta_ses
from roberta_ses.interface import Roberta_SES_Entailment


def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace

sys.settrace(trace)

ses = Roberta_SES_Entailment(
    roberta_path='/Users/ytshao/Desktop/Yutong/models/roberta-large-mnli/',
    ckpt_path='/Users/ytshao/Desktop/Yutong/external_repos/Roberta_SES/checkpoints/roberta-large/epoch=2-valid_loss=-0.2620-valid_acc_end=0.9223.ckpt',
    max_length=512,
    device_name='cpu')


pred = ses.predict(
    "I like sports",  # premise 
    "I like soccer",  # hypothesis 
)

print(pred)