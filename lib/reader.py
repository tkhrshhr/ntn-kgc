from collections import defaultdict

W_PATHs = ['data/wordnet-mlj12/wordnet-mlj12-train.txt',
           'data/wordnet-mlj12/wordnet-mlj12-valid.txt',
           'data/wordnet-mlj12/wordnet-mlj12-test.txt']

F_PATHs = ['data/FB15k/freebase_mtr100_mte100-train.txt',
           'data/FB15k/freebase_mtr100_mte100-valid.txt',
           'data/FB15k/freebase_mtr100_mte100-test.txt']

W11_PATHs = ['data/nips13-dataset/Wordnet/train.txt',
             'data/nips13-dataset/Wordnet/dev.txt',
             'data/nips13-dataset/Wordnet/test.txt']

F13_PATHs = ['data/nips13-dataset/Freebase/train.txt',
             'data/nips13-dataset/Freebase/dev.txt',
             'data/nips13-dataset/Freebase/test.txt']

PATH_DICT = {'w': W_PATHs,
             'f': F_PATHs,
             'w11': W11_PATHs,
             'f13': F13_PATHs}


ent_dict = {}
rel_dict = {}

rs2o = defaultdict(list)
ro2s = defaultdict(list)

r2s = []
r2o = []

train = []
dev = []
test = []


def get_dict(*files):
    ent_set = set()
    rel_set = set()
    for f in files:
        for line in f.readlines():
            s, r, o = line.split()[:3]
            ent_set.add(s)
            ent_set.add(o)
            rel_set.add(r)

    for i, ent in enumerate(list(ent_set)):
        ent_dict[ent] = i
    for i, rel in enumerate(list(rel_set)):
        rel_dict[rel] = i


def get_rso_as_ID(line):
    s, r, o = line.split()
    rs2o[(rel_dict[r], ent_dict[s])].append(ent_dict[o])
    ro2s[(rel_dict[r], ent_dict[o])].append(ent_dict[s])
    return [rel_dict[r], ent_dict[s], ent_dict[o]]


def get_rso_as_ID_w11f13_dev_test(line):
    s, r, o, label = line.split()
    rs2o[(rel_dict[r], ent_dict[s])].append(ent_dict[o])
    ro2s[(rel_dict[r], ent_dict[o])].append(ent_dict[s])
    label = int(label)
    if label == -1:
        label = 0
    return [rel_dict[r], ent_dict[s], ent_dict[o], int(label)]


def read(kg):
    train_f, dev_f, test_f = [open(path, 'r') for path in PATH_DICT[kg]]
    get_dict(train_f, dev_f, test_f)

    train_f.seek(0)
    dev_f.seek(0)
    test_f.seek(0)
    for line in train_f.readlines():
        train.append(get_rso_as_ID(line))

    # Create relation to subject or object dictionary
    r2s_ = defaultdict(set)
    r2o_ = defaultdict(set)
    for triplet in train:
        r, s, o = triplet
        r2s_[r].add(s)
        r2o_[r].add(o)

    for r in range(len(rel_dict)):
        r2s.append(list(r2s_[r]))
        r2o.append(list(r2o_[r]))

    if kg == ('w' or 'f'):
        func = get_rso_as_ID
    else:
        func = get_rso_as_ID_w11f13_dev_test
    for line in dev_f.readlines():
        dev.append(func(line))
    for line in test_f.readlines():
        test.append(func(line))

    return train, dev, test, len(ent_dict), len(rel_dict), rs2o, ro2s, r2s, r2o


if __name__ == '__main__':
    a, b, c, d, e, f, g, h, i = read('f13')
    print(b[:10])
    print(i[0])
    print(len(r2s))
    print(len(r2o))
