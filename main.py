import os
import time
import math

from srb_gpt import wiki, tokenizer, data, gpt, helper
import numpy as np
import torch

from torchsummary import summary


# wiki data source
BASE_URL = 'https://sr.wikipedia.org'
ROOT_LINK = 'https://sr.wikipedia.org/wiki/%D0%9D%D0%B8%D0%BA%D0%BE%D0%BB%D0%B0_%D0%A2%D0%B5%D1%81%D0%BB%D0%B0' # Nikola Tesla

# datafile
DATAFILE = 'data/data.txt'
BIN_DATAFILE = 'data/data.npy'

# tokenizer model file
TOKENIZER_DIR = 'models'
TOKENIZER_MODEL = f'{TOKENIZER_DIR}/regex.model'
OUR_SPLIT_PATTERN = r"""'|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
VOCAB_SIZE = 512

# model
BLOCK_SIZE = 128
N_LAYER = 2
N_HEAD = 4
N_EMBD = 128

# train
DEVICE = 'cpu'
BATCH_SIZE = 64
ITERS = 5000
MAX_LR = 6e-3
MIN_LR = MAX_LR / 10
WARMUP_ITERS = 200
LR_DECAY_DUR = 60000

WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.95

LOG_INTERVAL = 10

if __name__ == '__main__':
    # flags
    # state
    model_cfg = gpt.GPTConfig(block_size=BLOCK_SIZE, vocab_size=VOCAB_SIZE, n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=0.1, bias=False)
    # check if we need to download data
    if not os.path.exists(DATAFILE):
        print(f'DATAFILE: {DATAFILE} Not Found, downloading data')
        t0 = time.time()
        wiki.download_wiki_data_around_link(ROOT_LINK, BASE_URL, DATAFILE)
        t1 = time.time()
        print(f'DATAFILE: {DATAFILE} Created, took {t1 - t0:.2f} seconds')
    else:
        print(f'DATAFILE: {DATAFILE} Exists')

    # check if we have trained tokenizer
    tok = tokenizer.RegexTokenizer(OUR_SPLIT_PATTERN)
    if os.path.exists(TOKENIZER_MODEL):
        print(f'TOKENIZER: {TOKENIZER_MODEL} Loading pretrained')
        tok.load(TOKENIZER_MODEL)
    else:
        # train tokenizer
        print(f'TOKENIZER: {TOKENIZER_MODEL} Not Found, training it using DATAFILE: {DATAFILE}')
        text = open(DATAFILE, "r", encoding="utf-8").read()
        os.makedirs("models", exist_ok=True)
        t0 = time.time()
        tok.train(text, VOCAB_SIZE, verbose=True)
        t1 = time.time()
        tok.save(TOKENIZER_MODEL.split('.')[0])
        print(f'TOKENIZER: {TOKENIZER_MODEL} Trained, took {t1 - t0:.2f} seconds')

    # Is there bin of text data
    if not os.path.exists(BIN_DATAFILE):
        with open(DATAFILE, mode='r', encoding='utf-8') as file:
            text = file.read()

        print(f'ENCODING: {DATAFILE}')
        t0 = time.time()
        ids = tok.encode(text)
        t1 = time.time()
        print(f'ENCODING: {DATAFILE}, took {t1 - t0:.2f} seconds')
        arr = np.array(ids).astype(np.uint16)
        fp_save = np.memmap(BIN_DATAFILE, dtype='uint16', mode='w+', shape=(arr.shape[0],))
        fp_save[:] = arr[:]

    # Link BIN_DATAFILE and np, don't read it, just refit to memory
    fp = np.memmap(BIN_DATAFILE, dtype='uint16', mode='r')
    #TODO data split
    #TODO validation loop

    model = gpt.GPT(model_cfg).to(DEVICE)
    optimizer = model.configure_optimizers(WEIGHT_DECAY, MAX_LR, (BETA1, BETA2), DEVICE)
    t0 = time.time()
    dt = 0
    for it in range(ITERS):
        lr = helper.get_lr(it, WARMUP_ITERS, MAX_LR, LR_DECAY_DUR, MIN_LR)
        X, Y = data.get_batch(fp, BATCH_SIZE, BLOCK_SIZE)
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        optimizer.zero_grad()
        logits, loss = model.forward(X, Y)
        if loss is not None:
            loss.backward() 
            optimizer.step()

        t1 = time.time()
        dt += t1 - t0
        t0 = t1

        if it % LOG_INTERVAL == 0:
            if loss is not None:
                print(f"iter {it}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
            dt = 0 # reset delta time

    # generate text
    x = torch.stack([torch.from_numpy(np.array(tok.encode("уређај назван секундарни генератор")).astype(np.int64))]).to(DEVICE)
    txt = model.generate(x, 200)
    txt = list(txt.detach().cpu().numpy()[0])
    print(tok.decode(txt))