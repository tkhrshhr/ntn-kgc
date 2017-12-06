from collections import defaultdict

W_TRAIN_PATH = 'data/wordnet-mlj12/wordnet-mlj12-train.txt'
W_DEV_PATH = 'data/wordnet-mlj12/wordnet-mlj12-valid.txt'
W_TEST_PATH = 'data/wordnet-mlj12/wordnet-mlj12-test.txt'

F_TRAIN_PATH = 'data/FB15k/freebase_mtr100_mte100-train.txt'
F_DEV_PATH = 'data/FB15k/freebase_mtr100_mte100-valid.txt'
F_TEST_PATH = 'data/FB15k/freebase_mtr100_mte100-test.txt'

ent_dict = {}
rel_dict = {}

rs2o = defaultdict(list)
ro2s = defaultdict(list)

train = []
dev = []
test = []


def get_dict(*files):
    ent_list = []
    rel_list = []

    ent_set = set()
    rel_set = set()
    for f in files:
        for line in f.readlines():
            s, r, o = line.split()
            if s not in ent_set:
                ent_list.append(s)
                ent_set.add(s)
            if o not in ent_set:
                ent_list.append(o)
                ent_set.add(o)
            if r not in rel_set:
                rel_list.append(r)
                rel_set.add(r)

    for i, ent in enumerate(ent_list):
        ent_dict[ent] = i
    for i, rel in enumerate(rel_list):
        rel_dict[rel] = i


def get_rso_as_ID(line):
    s, r, o = line.split()
    rs2o[(rel_dict[r], ent_dict[s])].append(ent_dict[o])
    ro2s[(rel_dict[r], ent_dict[o])].append(ent_dict[s])
    return [rel_dict[r], ent_dict[s], ent_dict[o]]


def read(w_or_f='w'):
    if w_or_f == 'w':
        train_f = open(W_TRAIN_PATH, 'r')
        dev_f = open(W_DEV_PATH, 'r')
        test_f = open(W_TEST_PATH, 'r')
    elif w_or_f == 'f':
        train_f = open(F_TRAIN_PATH, 'r')
        dev_f = open(F_DEV_PATH, 'r')
        test_f = open(F_TEST_PATH, 'r')

    get_dict(train_f, dev_f, test_f)
    train_f.seek(0)
    dev_f.seek(0)
    test_f.seek(0)
    for line in train_f.readlines():
        train.append(get_rso_as_ID(line))
    for line in dev_f.readlines():
        dev.append(get_rso_as_ID(line))
    for line in test_f.readlines():
        test.append(get_rso_as_ID(line))

    return train, dev, test, len(ent_dict), len(rel_dict), rs2o, ro2s


if __name__ == '__main__':
    a, b, c, d, e, f, g = read('f')
    print(b[:10])
