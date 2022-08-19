def normalize_sentence(sen, token, max_length):
    n_padding = max_length - len(sen)
    return sen + [token] * n_padding if n_padding > 0 else sen[:max_length]

def format(text, vocab, v2, max_length):
    print(text)
    src = [vocab[s] for s in list(text)] + [vocab.end_token]
    tar = [v2.start_token] + [v2[s] for s in list("Il me faut mettre mes chaus")]

    return normalize_sentence(src, vocab.pad_token, max_length), tar