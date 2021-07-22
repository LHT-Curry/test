#coding=utf-8
import requests
import json
from tqdm import tqdm
from data_process import read_data


def call_evaluation(querys, labels, error=False):
    leakcall = miscall = n = 0.0
    error_dict = {}
    result_dict = {}
    for query, label in tqdm(zip(querys, labels)):
        url = "http://192.168.237.148:8901/debugsearch?query={}&business={}&messageid=1".format(query, business)
        res = requests.request("GET", url, headers={}, data={})
        res_json = res.json()
        intent = res_json['intent']
        if intent == label:
            n += 1
        elif not intent and label == 'QA-二类问题':
            n += 1
        else:
            error_dict[query + '_' + label] = res_json
        if label != 'QA-二类问题' and intent != label:
            if intent and intent != 'QA-二类问题':
                print("query:", query, "label:", label, 'intent', intent)
                miscall += 1
            else:
                leakcall += 1
        result_dict[query+'_'+label] = res_json
    acc = n / len(querys)
    if error:
        return result_dict, error_dict, acc, miscall, leakcall
    else:
        return result_dict, {}, acc, miscall, leakcall


def write_data(write_filename, write_data):
    with open('result/{}.json'.format(write_filename), 'w') as f:
        f.write(json.dumps(write_data, ensure_ascii=False))


if __name__ == '__main__':
    filename = 'test_50_data.file'
    write_filename = 'ndcgparser_idcg1_50'
    error_filename = 'ndcgparser_idcg1_50_error'
    business = 'lht_ndcg_50_test'
    querys, labels = read_data(filename)
    error = True
    result_dict, error_dict, acc, miscall, leakcall = call_evaluation(querys, labels, error)
    write_data(write_filename, result_dict)
    if error:
        write_data(error_filename, error_dict)
    print('acc:', acc)
    print('miscall:', miscall)
    print('leakcall:', leakcall)
