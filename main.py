import os
import time
import math

from srb_gpt import wiki, tokenizer, data, gpt
import numpy as np



# wiki data source
BASE_URL = 'https://sr.wikipedia.org'
ROOT_LINK = 'https://sr.wikipedia.org/wiki/%D0%9D%D0%B8%D0%BA%D0%BE%D0%BB%D0%B0_%D0%A2%D0%B5%D1%81%D0%BB%D0%B0' # Nikola Tesla

# datafile
DATAFILE = 'data.txt'
BIN_DATAFILE = 'data.npy'

# tokenizer model file
TOKENIZER_DIR = 'models'
TOKENIZER_MODEL = f'{TOKENIZER_DIR}/regex.model'
OUR_SPLIT_PATTERN = r"""'|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
VOCAB_SIZE = 512

# model
BLOCK_SIZE = 128
N_LAYER = 8
N_HEAD = 4
N_EMBD = 256

# train
DEVICE = 'cpu'
BATCH_SIZE = 16
ITERS = 2000
MIN_LR = 6e-5
WARMUP_ITERS = 200
MAX_LR = 6e-4
LR_DECAY_DUR = 60000

WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.95

LOG_INTERVAL = 10

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < WARMUP_ITERS:
        return MAX_LR * it / WARMUP_ITERS
    # 2) if it > lr_decay_iters, return min learning rate
    if it > LR_DECAY_DUR:
        return MIN_LR
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_DUR - WARMUP_ITERS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return MIN_LR + coeff * (MAX_LR - MIN_LR)


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
        text = open("cleaned_data.txt", "r", encoding="utf-8").read()
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
        np.save(BIN_DATAFILE, np.array(ids).astype(np.uint16))

    # Link BIN_DATAFILE and np, don't read it, just refit to memory
    fp = np.memmap(BIN_DATAFILE, dtype='uint16', mode='r')
    
    
    model = gpt.GPT(model_cfg)
    optimizer = model.configure_optimizers(WEIGHT_DECAY, MAX_LR, (BETA1, BETA2), DEVICE)
    t0 = time.time()
    for it in range(ITERS):
        lr = get_lr(it)
        X, Y = data.get_batch(fp, BATCH_SIZE, BLOCK_SIZE)
        optimizer.zero_grad()
        logits, loss = model.forward(X, Y)
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if it % LOG_INTERVAL == 0:
            if loss is not None:
                print(f"iter {it}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")