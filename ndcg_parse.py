#coding=utf-8
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NdcgParser(object):
    def __init__(self, data):
        self.data = data
        self.query = []
        self.label = []
        self.gap_score = []
        self.gap_mean_score = []
        self.gap_score_miscall = []
        self.gap_mean_score_miscall = []
        self.mean = []
        self.ndcg = []
        self.mean_miscall = []
        self.ndcg_miscall = []
        self.num_miscall = self.num_correct = 0
        self.positive_score = []
        self.negative_score = []

    def pltscatter(self, correct_datax, correct_datay, miscall_datax, miscall_datay, figname, xlabel, ylabel):
        plt.scatter(correct_datax, correct_datay, c='b', label='correct')
        plt.scatter(miscall_datax, miscall_datay, c='r', label='miscall')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(figname)
        plt.show()

    def plthist(self, figname, data):
        plt.hist(data)
        plt.savefig(figname)
        plt.show()

    def percentile(self, score):
        p_min = min(score)
        p_max = max(score)
        i = 5
        p = [p_min]
        while i < 100:
            p.append(np.percentile(score, i))
            i += 5
        p.append(p_max)
        p = np.around(np.array(p), 4)
        return p

    def data_analysis(self):
        for key in self.data:
            try:
                query, label = key.split('_')
                intent = self.data[key]['intent']
                if self.data[key]['debug']['parser']["sepical_info"]['ndcg_score_messg']["ndcg_rerank_result"]:
                    ndcg_rerank_result = self.data[key]['debug']['parser']["sepical_info"]['ndcg_score_messg'][
                        "ndcg_rerank_result"]
                    ndcg_rerank_result.sort(key=lambda i: i["score"], reverse=True)
                    for result in ndcg_rerank_result:
                        node_name = result['node_name']
                        score = result['score']
                        if node_name == label:
                            self.positive_score.append(score)
                        else:
                            self.negative_score.append(score)
                    if len(ndcg_rerank_result) >= 2:
                        score1 = ndcg_rerank_result[0]['score']
                        score2 = ndcg_rerank_result[1]['score']
                        mean1 = ndcg_rerank_result[0]['mean']
                        mean2 = ndcg_rerank_result[1]['mean']
                        if intent == label and label != 'QA-二类问题':
                            self.gap_score.append(score1 - score2)
                            self.gap_mean_score.append(mean1 - mean2)
                            self.mean.append(mean1)
                            self.ndcg.append(score1)
                            self.num_correct += 1
                        if label != 'QA-二类问题' and intent != label:
                            if intent and intent != 'QA-二类问题':
                                self.mean_miscall.append(mean1)
                                self.ndcg_miscall.append(score1)
                                self.gap_score_miscall.append(score1 - score2)
                                self.gap_mean_score_miscall.append(mean1 - mean2)
                                self.num_miscall += 1
            except:
                pass

    def get_result(self, ndcg_mean_gap_figname, ndcg_gap_figname, positve_figname, negative_figname):
        self.pltscatter(self.gap_score, self.gap_mean_score, self.gap_score_miscall, self.gap_mean_score_miscall,
                   'figure/{}'.format(ndcg_mean_gap_figname), 'ndcg_gap', 'mean_gap')
        self.pltscatter(self.ndcg, self.gap_score, self.ndcg_miscall, self.gap_score_miscall,
                        'figure/{}'.format(ndcg_gap_figname), 'ndcg', 'ndcg_gap')
        print('num_correct:', self.num_correct)
        print('num_miscall:', self.num_miscall)
        p_ndcg = self.percentile(self.ndcg)
        p_ndcg_miscall = self.percentile(self.ndcg_miscall)
        df_ndcg = {'百分位': ['min'] + [str(i) for i in range(5, 100, 5)] + ['max'],
                   '正确样本': p_ndcg,
                   '错误样本': p_ndcg_miscall}
        df_ndcg = pd.DataFrame(df_ndcg).T
        df_ndcg.to_csv('percent_result/ndcg.csv', header=False)
        p_ndcg_gap = self.percentile(self.gap_score)
        p_ndcg_gap_miscall = self.percentile(self.gap_score_miscall)
        df_ndcg_gap = {'百分位': ['min'] + [str(i) for i in range(5, 100, 5)] + ['max'],
                   '正确样本': p_ndcg_gap,
                   '错误样本': p_ndcg_gap_miscall}
        df_ndcg_gap = pd.DataFrame(df_ndcg).T
        df_ndcg_gap.to_csv('percent_result/ndcg_gap.csv', header=False)
        p_postive = self.percentile(self.positive_score)
        p_negative = self.percentile(self.negative_score)
        df_pos_nege = {'百分位': ['min'] + [str(i) for i in range(5, 100, 5)] + ['max'],
                   '正样本': p_postive,
                   '负样本': p_negative}
        df_pos_nege = pd.DataFrame(df_pos_nege).T
        df_pos_nege.to_csv('percent_result/pos_nege.csv', header=False)
        self.plthist('figure/{}'.format(positve_figname), self.positive_score)
        self.plthist('figure/{}'.format(negative_figname), self.negative_score)


if __name__ == '__main__':
    resultname = 'ndcgparser_idcg1_50'
    ndcg_mean_gap_figname = 'ndcg_mean_gap'
    ndcg_gap_figname = 'ndcg_gap'
    positve_figname = 'positive'
    negative_figname = 'negative'
    with open('result/{}.json'.format(resultname), 'r') as f:
        data = json.load(f)
    parser = NdcgParser(data)
    parser.data_analysis()
    parser.get_result(ndcg_mean_gap_figname, ndcg_gap_figname, positve_figname, negative_figname)