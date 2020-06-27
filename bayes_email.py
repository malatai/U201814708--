import numpy as np


def load_data_set(filename):
    """读取数据集"""
    data_mat = []
    list_class = []
    fr = open(filename)
    for line in fr.readlines():
        cur_line = line.strip().split(',')
        data_mat.append(cur_line)
        list_class.append(cur_line[-1])
    return data_mat, list_class


def create_vocab_list(data_set):
    """创建所有词汇的集合"""
    vocab_set = set()
    for doc in data_set:
        vocab_set = vocab_set | set(doc)
    return list(vocab_set)


def create_vocab_vector(vocab_list, input_vocab_list):
    """创建输入词汇的向量"""

    vector = [0] * len(vocab_list)
    index = 0
    for word in vocab_list:
        if word in input_vocab_list:
            vector[index] += 1
        index += 1
    return vector


def native_bayes_0(train_matrix, list_classes):
    """
    :param train_matrix: 训练样本矩阵 由多个文档的词汇向量组成
    :param list_classes: 训练样本分类向量
    :return:p_1_class 任一文档分类为1的概率  p_0_vect,p_1_vect 分别为给定类别的情况下单词的出现概率
    """
    # 训练样本个数
    num_train_docs = len(train_matrix)
    # 词汇总数量
    num_words = len(train_matrix[0])
    # 分类为1的样本占比
    p_1_class = sum(list_classes) / float(num_train_docs)

    # 分类为 0 或 1 的所有样本的每个词汇出现的个数
    p_0_num = np.ones(num_words)
    p_1_num = np.ones(num_words)
    p_0_count = 2.0
    p_1_count = 2.0

    for i in range(num_train_docs):
        # 该训练样本分类为1
        if list_classes[i] == 1:
            p_1_num += train_matrix[i]
            p_0_count += sum(train_matrix[i])
        else:
            p_0_num += train_matrix[i]
            p_1_count += sum(train_matrix[i])

    # 避免概率成绩过小四舍五入为0，这里取对数
    p_1_vect = np.log(p_1_num / p_1_count)
    p_0_vect = np.log(p_0_num / p_0_count)
    return p_0_vect, p_1_vect, p_1_class


def classify_bayes(test_vector, p_0_vector, p_1_vector, p_1_class):
    """

    :param test_vector: 要分类的测试向量
    :param p_0_vector: 给定类别为0的情况下单词的出现概率
    :param p_1_vector: 给定类别为1的情况下单词的出现概率
    :param p_1_class: 任一文档分类为1的概率
    :return:
    """
    # 计算每个分类的概率(概率相乘取对数 = 概率各自对数相加)
    p1 = sum(test_vector * p_1_vector) + np.log(p_1_class)
    p0 = sum(test_vector * p_0_vector) + np.log(1.0 - p_1_class)

    if p1 > p0:
        return 1
    else:
        return 0


def test_bayes():
    file_name = ''
    data_set, list_class = load_data_set(file_name)

    vocab_list = create_vocab_list(data_set)
    train_data_mat = []
    for doc in data_set:
        vector = create_vocab_vector(vocab_list, doc)
        train_data_mat.append(vector)
    p_0_v, p_1_v, p_1_c = native_bayes_0(np.array(train_data_mat), np.array(list_class))

    test_doc = []
    test_vector = create_vocab_vector(vocab_list, test_doc)
    print(test_doc, '分类是：', classify_bayes(test_vector, p_0_v, p_1_v, p_1_c))
test_bayes()