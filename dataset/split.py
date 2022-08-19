from sklearn.model_selection import train_test_split

data = None
with open('./fra-eng/fra.txt') as f:
    data = f.read().splitlines()
    train, test = train_test_split(data, test_size = 0.07)

with open('./train.txt', 'w') as fw:
    fw.write('\n'.join(train))

with open('./val.txt', 'w') as fw:
    fw.write('\n'.join(test))