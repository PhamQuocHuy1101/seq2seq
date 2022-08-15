def normalize_sentence(sen, token, max_length):
    n_padding = max_length - len(sen)
    return sen + [token] * n_padding if n_padding > 0 else sen[:max_length]