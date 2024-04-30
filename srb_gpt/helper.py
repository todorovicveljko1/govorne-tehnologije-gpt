import math

def get_lr(it, WARMUP_ITERS, MAX_LR, LR_DECAY_DUR, MIN_LR, ):
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