import numpy as np
import os
import network
from utils import pos_embed


# 训练集不能有空行，不然会被截断！！！！！！！！！！！！！！！！！

# find the index of x in y, if x not in y, return -1
def find_index(x, y):
    flag = -1
    for i in range(len(y)):
        if x != y[i]:
            continue
        else:
            return i
    return flag


# reading data
def init():
    print('reading word embedding data...')
    vec = []
    word2id = {}
    f = open('./origin_data/vec.txt', encoding='utf-8')
    content = f.readline()
    content = content.strip().split()
    dim = int(content[1])
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

    print('reading relation to id')
    relation2id = {}
    f = open('./origin_data/relation2id.txt', 'r', encoding='utf-8')
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        relation2id[content[0]] = int(content[1])
    f.close()

    # length of sentence is 200
    max_sen_len = network.Settings();
    # max length of position embedding is 60 (-60~+60)
    # maxlen = 60


    train_sen = {}  # {entity pair:[[[label1-sentence 1],[label1-sentence 2]...],[[label2-sentence 1],[label2-sentence 2]...],...]}
    train_ans = {}  # {entity pair:[label1,label2,...]} the label is one-hot vector

    print('reading train data...')
    f = open('./origin_data/train.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '' or content == '\n':
            break

        content = content.strip().split()
        # get entity name
        en1 = content[0]
        en2 = content[1]
        relation = 0
        if content[2] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[2]]
        # put the same entity pair sentences into a dict
        tup = (en1, en2)
        if tup not in train_sen:
            train_sen[tup] = []
            train_sen[tup].append([])
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            train_ans[tup] = []
            train_ans[tup].append(label)
        else:
            y_id = relation

            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1

            temp = find_index(label, train_ans[tup])
            if temp == -1:
                train_ans[tup].append(label)
                label_tag = len(train_ans[tup]) - 1
                train_sen[tup].append([])
            else:
                label_tag = temp

        sentence = content[3]

        # 编码entity的位置
        # For Chinese
        en1pos = sentence.find(en1)
        if en1pos == -1:
            en1pos = 0
        en2pos = sentence.find(en2)
        if en2pos == -1:
            en2pos = 0
        
        output = []

        # Embedding the position
        for i in range(max_sen_len):
            word_id = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word_id, rel_e1, rel_e2])

        for i in range(min(max_sen_len, len(sentence))):
            word_id = 0
            if sentence[i] not in word2id:
                word_id = word2id['UNK']
            else:
                word_id = word2id[sentence[i]]
            output[i][0] = word_id # 一句话的list，list中的每个元素是[字id，相对实体1首字符的位置，相对实体2首字符的位置]

        train_sen[tup][label_tag].append(output)

    print('reading test data ...')

    test_sen = {}  # {entity pair:[[sentence 1],[sentence 2]...]}
    test_ans = {}  # {entity pair:[labels,...]} the labels is N-hot vector (N is the number of multi-label)

    f = open('./origin_data/dev.txt', 'r', encoding='utf-8')

    while True:
        content = f.readline()
        if content == '' or content == '\n':
            break

        content = content.strip().split()
        en1 = content[0]
        en2 = content[1]
        relation = 0
        if content[2] not in relation2id:
            relation = relation2id['NA']
        else:
            relation = relation2id[content[2]]
        tup = (en1, en2)

        if tup not in test_sen:
            test_sen[tup] = []
            y_id = relation
            label_tag = 0
            label = [0 for i in range(len(relation2id))]
            label[y_id] = 1
            test_ans[tup] = label
        else:
            y_id = relation
            test_ans[tup][y_id] = 1

        sentence = content[3]

        en1pos = 0
        en2pos = 0
        
        #For Chinese
        en1pos = sentence.find(en1)
        if en1pos == -1:
            en1pos = 0
        en2pos = sentence.find(en2)
        if en2pos == -1:
            en2post = 0
            
        output = []

        for i in range(max_sen_len):
            word_id = word2id['BLANK']
            rel_e1 = pos_embed(i - en1pos)
            rel_e2 = pos_embed(i - en2pos)
            output.append([word_id, rel_e1, rel_e2])

        for i in range(min(max_sen_len, len(sentence))):
            word_id = 0
            if sentence[i] not in word2id:
                word_id = word2id['UNK']
            else:
                word_id = word2id[sentence[i]]

            output[i][0] = word_id
        test_sen[tup].append(output)

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    print('organizing train data')
    f = open('./origin_data/train.txt', 'w', encoding='utf-8')
    temp = 0
    for i in train_sen: # i是tuple
        if len(train_ans[i]) != len(train_sen[i]):
            print('ERROR')
        re_num = len(train_ans[i]) # 每一对entity pair之间可能出现的关系数目
        for j in range(re_num):
            train_x.append(train_sen[i][j])
            train_y.append(train_ans[i][j])
            # f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + str(np.argmax(train_ans[i][j])) + '\n')
            temp += 1
    f.close()

    print('organizing dev data')
    f = open('./origin_data/dev.txt', 'w', encoding='utf-8')
    temp = 0
    for i in test_sen:
        test_x.append(test_sen[i])
        test_y.append(test_ans[i])
        tempstr = ''
        for j in range(len(test_ans[i])):
            if test_ans[i][j] != 0:
                tempstr = tempstr + str(j) + '\t'
        # f.write(str(temp) + '\t' + i[0] + '\t' + i[1] + '\t' + tempstr + '\n')
        temp += 1
    f.close()

    train_x = np.array(train_x) # train_x的列数是tuple * 该tuple下label的数目
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    np.save('./data/vec.npy', vec)
    np.save('./data/train_x.npy', train_x)
    np.save('./data/train_y.npy', train_y)
    np.save('./data/dev_x.npy', test_x)
    np.save('./data/dev_y.npy', test_y)

   


def seperate():
    print('reading training data')
    x_train = np.load('./data/train_x.npy')

    train_word = []
    train_pos1 = []
    train_pos2 = []

    print('seprating train data')
    for i in range(len(x_train)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_train[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        train_word.append(word)
        train_pos1.append(pos1)
        train_pos2.append(pos2)

    train_word = np.array(train_word) # 各tuple下，该tuple的各label下的句子的字的id
    train_pos1 = np.array(train_pos1)
    train_pos2 = np.array(train_pos2)
    np.save('./data/train_word.npy', train_word)
    np.save('./data/train_pos1.npy', train_pos1)
    np.save('./data/train_pos2.npy', train_pos2)

    print('seperating dev data')
    x_test = np.load('./data/dev_x.npy')
    test_word = []
    test_pos1 = []
    test_pos2 = []

    for i in range(len(x_test)):
        word = []
        pos1 = []
        pos2 = []
        for j in x_test[i]:
            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            for k in j:
                temp_word.append(k[0])
                temp_pos1.append(k[1])
                temp_pos2.append(k[2])
            word.append(temp_word)
            pos1.append(temp_pos1)
            pos2.append(temp_pos2)
        test_word.append(word)
        test_pos1.append(pos1)
        test_pos2.append(pos2)

    test_word = np.array(test_word)
    test_pos1 = np.array(test_pos1)
    test_pos2 = np.array(test_pos2)

    np.save('./data/dev_word.npy', test_word)
    np.save('./data/dev_pos1.npy', test_pos1)
    np.save('./data/dev_pos2.npy', test_pos2)



# get answer metric for PR curve evaluation
def getans():
    test_y = np.load('./data/testall_y.npy')
    eval_y = []
    for i in test_y:
        eval_y.append(i[1:])
    allans = np.reshape(eval_y, (-1))
    np.save('./data/allans.npy', allans)


def get_metadata():
    fwrite = open('./data/metadata.tsv', 'w', encoding='utf-8')
    f = open('./origin_data/vec.txt', encoding='utf-8')
    f.readline()
    while True:
        content = f.readline().strip()
        if content == '':
            break
        name = content.split()[0]
        fwrite.write(name + '\n')
    f.close()
    fwrite.close()

def initialize():
    init()
    seperate()
    getans()
    get_metadata()

if __name__ == "__main__":
   initialize()
