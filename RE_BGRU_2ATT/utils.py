import logging
import numpy as np

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def load_relation():
    print('reading relation to id')
    relation2id = {}
    id2relation = {}
    f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
        id2relation[int(content[1])] = content[0]
    f.close()
    return relation2id, id2relation


def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return b

max_rel_pos = 60;

# embedding the position，可改进
def pos_embed(x):
    if x < -max_rel_pos:
        return 0
    if -max_rel_pos <= x <= max_rel_pos:
        return x + max_rel_pos+1
    if x > max_rel_pos:
        return 2 * max_rel_pos+2
