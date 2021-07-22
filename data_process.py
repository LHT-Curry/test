import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_data(filename):
    labels = []
    querys = []
    with open("data/{}".format(filename), 'r') as f:
        for line in f:
            label, query = line.replace('\n', '').split('\t')
            labels.append(label)
            querys.append(query)
    return querys, labels


def split_data(querys, labels, write_filename, testsize, nomean=False):
    data = pd.DataFrame(columns=['label', 'query'])
    data['label'] = labels
    data['query'] = querys
    a = data['label'].value_counts()
    data = data[data['label'] != 'QA-资料包邮吗']
    if nomean:
        data = data[data['label'] != 'QA-二类问题']
    data.drop_duplicates(subset=['query', 'label'], keep='first', inplace=True)
    a = data['label'].value_counts()
    # b = data.groupby('query').count()>1
    # q = b[b['label'] == True].index
    # repeat_df = data[data['query'].isin(q)]
    # print(1)
    train, test = train_test_split(data, test_size=testsize, stratify=data['label'], random_state=1)
    # train, dev = train_test_split(train, test_size=1/9, stratify=train['label'], random_state=1)
    train.to_csv('data/train_{}_data.file'.format(write_filename), sep='\t', index=False, header=False)
    test.to_csv('data/test_{}_data.file'.format(write_filename), sep='\t', index=False, header=False)
    # dev.to_csv('data/dev_{}_data.file'.format(write_filename), sep='\t', index=False, header=False)


def extract_data(querys, labels, write_filename, num, nomean=False):
    data = pd.DataFrame(columns=['label', 'query'])
    data['label'] = labels
    data['query'] = querys
    data = data[data['label'] != 'QA-资料包邮吗']
    if nomean:
        data = data[data['label'] != 'QA-二类问题']
    train = data.groupby('label').apply(lambda x: x.sample(frac=0.2)[:num])
    # a = train['label'].value_counts()
    train_query = list(train['query'])
    test = data[~(data['query'].isin(train_query))]
    # b = test['label'].value_counts()
    train.to_csv('data/train_{}_data.file'.format(write_filename), sep='\t', index=False, header=False)
    test.to_csv('data/test_{}_data.file'.format(write_filename), sep='\t', index=False, header=False)


if __name__ == '__main__':
    filename = 'input_data.txt'
    write_filename = '50_nomean'
    querys, labels = read_data(filename)
    # split_data(querys, labels, write_filename, 0.5, True)
    extract_data(querys, labels, write_filename, 50, True)
