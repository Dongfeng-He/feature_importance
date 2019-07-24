import pandas as pd
import numpy as np
import math
from scipy.sparse import csr_matrix
import re


def chi_square_bucketing(feature_list, label_list, confidence_val=3.841, max_bucket_num=10, sample_num=None):
    """
    卡方分桶
    feature_list: 需要卡方分箱的变量列表
    label_list：正负标签列表
    confidence_val：置信度水平（默认是不进行抽样95%）
    max_bucket_num：最多箱的数目
    sample_num: 为抽样的数目（默认是不进行抽样），因为如果观测值过多运行会较慢
    """
    data = pd.DataFrame({'feature_list': feature_list, 'label_list': label_list})
    df = data[['feature_list', 'label_list']]
    # 进行是否抽样操作
    if sample_num:
        df = df.sample(n=sample_num)
    # 进行数据格式化录入
    total_num = df.groupby(['feature_list'])['label_list'].count()  # 统计需分箱变量每个值数目
    total_num = pd.DataFrame({'total_num': total_num})  # 创建一个数据框保存之前的结果
    positive_class = df.groupby(['feature_list'])['label_list'].sum()  # 统计需分箱变量每个值正样本数
    positive_class = pd.DataFrame({'positive_class': positive_class})  # 创建一个数据框保存之前的结果
    regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True, how='inner')  # 组合total_num与positive_class
    regroup.reset_index(inplace=True)
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class']  # 统计需分箱变量每个值负样本数
    regroup = regroup.drop('total_num', axis=1)
    np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
    print('已完成数据读入,正在计算数据初处理')

    # 处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    i = 0
    while i <= np_regroup.shape[0] - 2:
        if (np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (
                np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 正样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 负样本
            np_regroup[i, 0] = np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1

    # 对相邻两个区间进行卡方值计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = (np_regroup[i, 1] * np_regroup[i + 1, 2] - np_regroup[i, 2] * np_regroup[i + 1, 1]) ** 2 \
              * (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) / \
              ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i + 1, 1] + np_regroup[i + 1, 2]) * (
                      np_regroup[i, 1] + np_regroup[i + 1, 1]) * (np_regroup[i, 2] + np_regroup[i + 1, 2]))
        chi_table = np.append(chi_table, chi)
    print('已完成数据初处理，正在进行卡方分箱核心操作')

    # 把卡方值最小的两个区间进行合并（卡方分箱核心）
    while True:
        if len(chi_table) <= (max_bucket_num - 1) and min(chi_table) >= confidence_val:
            break
        chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
        np_regroup[chi_min_index, 0] = np_regroup[chi_min_index + 1, 0]
        np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

        if chi_min_index == np_regroup.shape[0] - 1:  # 最小值试最后两个区间的时候
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                            np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (
                                                       np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                        np_regroup[chi_min_index - 1, 1] + np_regroup[
                                                    chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[
                                               chi_min_index, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index, axis=0)
        else:
            # 计算合并后当前区间与前一个区间的卡方值并替换
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index - 1, 1] * np_regroup[chi_min_index, 2] -
                                            np_regroup[chi_min_index - 1, 2] * np_regroup[chi_min_index, 1]) ** 2 \
                                           * (np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2] +
                                              np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                                           ((np_regroup[chi_min_index - 1, 1] + np_regroup[chi_min_index - 1, 2]) * (
                                                       np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                        np_regroup[chi_min_index - 1, 1] + np_regroup[
                                                    chi_min_index, 1]) * (np_regroup[chi_min_index - 1, 2] + np_regroup[
                                               chi_min_index, 2]))
            # 计算合并后当前区间与后一个区间的卡方值并替换
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index + 1, 2] - np_regroup[
                chi_min_index, 2] * np_regroup[chi_min_index + 1, 1]) ** 2 \
                                       * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[
                chi_min_index + 1, 1] + np_regroup[chi_min_index + 1, 2]) / \
                                       ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (
                                                   np_regroup[chi_min_index + 1, 1] + np_regroup[
                                               chi_min_index + 1, 2]) * (
                                                    np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]) * (
                                                    np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]))
            # 删除替换前的卡方值
            chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)
    print('已完成卡方分箱核心操作，正在保存结果')

    # 把结果保存成一个数据框
    result_data = pd.DataFrame()  # 创建一个保存结果的数据框
    list_temp = []
    for i in np.arange(np_regroup.shape[0]):
        if i == 0:
            x = "[-∞, %s]" % str(np_regroup[i, 0])
        elif i == np_regroup.shape[0] - 1:
            x = "[%s, +∞]" % str(np_regroup[i - 1, 0])
        else:
            x = "[%s, %s]" % (str(np_regroup[i - 1, 0]), str(np_regroup[i, 0]))
        list_temp.append(x)
    result_data['interval'] = list_temp  # 结果表第二列：区间
    result_data['negative_num'] = np_regroup[:, 2]  # 结果表第三列：负样本数目
    result_data['positive_num'] = np_regroup[:, 1]  # 结果表第四列：正样本数目
    bucket_list = np_regroup[:, 0].tolist()
    return result_data, bucket_list


def even_num_bucketing(feature_list, bucket_num):
    """
    等频分桶
    """
    sorted_list = sorted(feature_list)
    avg_bucket_sample_num = len(feature_list) / bucket_num
    bucket_list = [sorted_list[math.floor(avg_bucket_sample_num * i)] for i in range(1, bucket_num)]
    """
    bucket_set = set()
    bucket_list = []
    for bucket in tmp_bucket_list:
        if bucket not in bucket_set:
            bucket_set.add(bucket)
            bucket_list.append(bucket)
    """
    return bucket_list


def create_bucket_name_dict(bucket_list, name):
    """
    创建每个桶对应变量区间名字典
    """
    bucket_name_dict = {}
    for i in range(len(bucket_list) + 1):
        if i == 0:
            bucket_name = "%s[0, %s)" % (name, bucket_list[i])
        elif i == len(bucket_list):
            bucket_name = "%s[%s, +∞)" % (name, bucket_list[i - 1])
        else:
            bucket_name = "%s[%s, %s)" % (name, bucket_list[i - 1], bucket_list[i])
        bucket_name_dict[i] = bucket_name
    return bucket_name_dict


def feature_discretization(feature_list, bucket_list, one_hot=False):
    """
    根据分桶进行离散化
    """
    cate_list = []
    bucket_list = sorted(bucket_list)
    for feature in feature_list:
        cate = 0
        for bucket in bucket_list:
            if feature >= bucket:
                cate += 1
            else:
                break
        cate_list.append(cate)
    if one_hot:
        dim = len(bucket_list) + 1
        one_hot_list = []
        for cate in cate_list:
            vec = [0] * dim
            vec[cate] = 1
            one_hot_list.append(vec)
        return one_hot_list
    else:
        return cate_list


def feature_cross_2(feature_comb1, feature_comb2, one_hot=False):
    """
    二阶特征交叉
    """
    feature_list1 = feature_comb1[0]
    feature_list2 = feature_comb2[0]
    bucket_name_dict1 = feature_comb1[1]
    bucket_name_dict2 = feature_comb2[1]
    feature_num1 = max(feature_list1) + 1
    feature_num2 = max(feature_list2) + 1
    cross_dim = feature_num1 * feature_num2
    cross_index_dict = {}
    index_name_dict = {}
    index = 0
    for cate1 in range(feature_num1):
        for cate2 in range(feature_num2):
            cross_feature = "%d-%d" % (cate1, cate2)
            cross_feature_name = "%s %s" % (bucket_name_dict1[cate1], bucket_name_dict2[cate2])
            cross_index_dict[cross_feature] = index
            index_name_dict[index] = cross_feature_name
            index += 1
    cross_feature_list = []
    for feature1, feature2 in zip(feature_list1, feature_list2):
        index = cross_index_dict[str(feature1) + "-" + str(feature2)]
        if one_hot:
            vec = [0] * cross_dim
            vec[index] = 1
            cross_feature_list.append(vec)
        else:
            cross_feature_list.append(index)
    return cross_feature_list, index_name_dict


def feature_cross_3(feature_comb1, feature_comb2, feature_comb3, one_hot=False):
    """
    三阶特征交叉
    """
    feature_list1 = feature_comb1[0]
    feature_list2 = feature_comb2[0]
    feature_list3 = feature_comb3[0]
    bucket_name_dict1 = feature_comb1[1]
    bucket_name_dict2 = feature_comb2[1]
    bucket_name_dict3 = feature_comb3[1]
    feature_num1 = max(feature_list1) + 1
    feature_num2 = max(feature_list2) + 1
    feature_num3 = max(feature_list3) + 1
    cross_dim = feature_num1 * feature_num2 * feature_num3
    cross_index_dict = {}
    index_name_dict = {}
    index = 0
    for cate1 in range(feature_num1):
        for cate2 in range(feature_num2):
            for cate3 in range(feature_num3):
                cross_feature = "%d-%d-%d" % (cate1, cate2, cate3)
                cross_feature_name = "%s %s %s" % (bucket_name_dict1[cate1], bucket_name_dict2[cate2], bucket_name_dict3[cate3])
                cross_index_dict[cross_feature] = index
                index_name_dict[index] = cross_feature_name
                index += 1
    cross_feature_list = []
    for feature1, feature2, feature3 in zip(feature_list1, feature_list2, feature_list3):
        index = cross_index_dict[str(feature1) + "-" + str(feature2) + "-" + str(feature3)]
        if one_hot:
            vec = [0] * cross_dim
            vec[index] = 1
            cross_feature_list.append(vec)
        else:
            cross_feature_list.append(index)
    return cross_feature_list, index_name_dict


def compute_retention_rate(feature_comb_list, label_list):
    """
    计算区间留存率
    """
    overall_bucket_name_dict = {}
    add_up_list = []
    for feature_comb in feature_comb_list:
        add_up = len(overall_bucket_name_dict)
        add_up_list.append(add_up)
        bucket_name_dict = feature_comb[1]
        for index, bucket_name in bucket_name_dict.items():
            overall_bucket_name_dict[index + add_up] = bucket_name
    dim = len(overall_bucket_name_dict)
    retention_list = [[0, 0] for _ in range(dim)]
    retention_rate_list = []
    proportion_list = []
    for i in range(len(feature_comb_list[0][0])):
        label = label_list[i]
        for j in range(len(feature_comb_list)):
            index = feature_comb_list[j][0][i] + add_up_list[j]
            retention_list[index][label] += 1
    for feature in range(dim):
        positive_num = retention_list[feature][1]
        total_num = sum(retention_list[feature])
        retention_rate = positive_num / total_num if total_num > 0 else 0
        proportion = total_num / len(label_list)
        retention_rate_list.append(retention_rate)
        proportion_list.append(proportion)
    retention_dict = {"retention_rate_list": retention_rate_list, "proportion_list": proportion_list}
    return retention_dict


def feature_concat(feature_comb_list):
    """
    特征拼接
    """
    overall_bucket_name_dict = {}
    add_up_list = []
    for feature_comb in feature_comb_list:
        add_up = len(overall_bucket_name_dict)
        add_up_list.append(add_up)
        bucket_name_dict = feature_comb[1]
        for index, bucket_name in bucket_name_dict.items():
            overall_bucket_name_dict[index + add_up] = bucket_name
    dim = len(overall_bucket_name_dict)
    sample_list = []
    for i in range(len(feature_comb_list[0][0])):
        vec = [0] * dim
        for j in range(len(feature_comb_list)):
            index = feature_comb_list[j][0][i] + add_up_list[j]
            vec[index] = 1
        sample_list.append(vec)
    return sample_list, overall_bucket_name_dict


def feature_concat_dense(feature_comb_list):
    """
    特征拼接，连续
    """
    overall_bucket_name_dict = {}
    add_up_list = []
    for feature_comb in feature_comb_list:
        add_up = len(overall_bucket_name_dict)
        add_up_list.append(add_up)
        bucket_name_dict = feature_comb[1]
        for index, bucket_name in bucket_name_dict.items():
            overall_bucket_name_dict[index + add_up] = bucket_name
    sample_list = []
    for i in range(len(feature_comb_list[0][0])):
        sample = []
        for j in range(len(feature_comb_list)):
            sample.append(feature_comb_list[j][0][i])
        sample_list.append(sample)
    return sample_list, overall_bucket_name_dict


def feature_concat_sparse(feature_comb_list, train_num):
    """
    特征拼接，onehot
    """
    overall_bucket_name_dict = {}
    add_up_list = []
    for feature_comb in feature_comb_list:
        add_up = len(overall_bucket_name_dict)
        add_up_list.append(add_up)
        bucket_name_dict = feature_comb[1]
        for index, bucket_name in bucket_name_dict.items():
            overall_bucket_name_dict[index + add_up] = bucket_name
    dim = len(overall_bucket_name_dict)
    train_row = []
    train_col = []
    train_data = []
    valid_row = []
    valid_col = []
    valid_data = []
    for i in range(len(feature_comb_list[0][0])):
        for j in range(len(feature_comb_list)):
            index = feature_comb_list[j][0][i] + add_up_list[j]
            if i < train_num:
                train_row.append(i)
                train_col.append(index)
                train_data.append(1)
            else:
                valid_row.append(i - train_num)
                valid_col.append(index)
                valid_data.append(1)
    train_csr = csr_matrix((np.array(train_data), (np.array(train_row), np.array(train_col))), shape=(train_num, dim))
    valid_csr = csr_matrix((np.array(valid_data), (np.array(valid_row), np.array(valid_col))), shape=(len(feature_comb_list[0][0]) - train_num, dim))
    return train_csr, valid_csr, overall_bucket_name_dict


# 测试
if __name__ == "__main__":
    # feature_list = [12, 23, -10, -9, 23, 1, 3, 23, 2, 35, 2, 3235, 573, 123, 5, 46, 46, 4123]
    feature_list1 = [-12, -23, -1, -2, -23, -1, -3, -23, -2, -35, -2, -3235, -573, -123, -5, -46, -46, -4123]
    label_list1 = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    feature_list2 = [12, 23, -10, -9, 23, 1, 3, 23, 2, 35, 2, 3235, 573, 123, 5, 0, 2, 3]
    label_list2 = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    feature_list3 = [12, 2345, -123, 9, 23, 1, -3, 23, 2, -35, 2, 23, 573, 35, 5, 0, 2, 3]
    label_list3 = [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]
    # confidenceVal=3.841
    result1, bucket_list1 = chi_square_bucketing(feature_list1, label_list1, confidence_val=3, max_bucket_num=10, sample_num=None)
    result2, bucket_list2 = chi_square_bucketing(feature_list2, label_list2, confidence_val=3, max_bucket_num=10, sample_num=None)
    result3, bucket_list3 = chi_square_bucketing(feature_list2, label_list2, confidence_val=3, max_bucket_num=10, sample_num=None)

    #bucket_list = even_num_bucketing(feature_list, bucket_num=3)
    bucket_name_dict1 = create_bucket_name_dict(bucket_list1, "气温")
    discrete_feature_list1 = feature_discretization(feature_list1, bucket_list1)
    bucket_name_dict2 = create_bucket_name_dict(bucket_list2, "深度")
    discrete_feature_list2 = feature_discretization(feature_list2, bucket_list2)
    bucket_name_dict3 = create_bucket_name_dict(bucket_list3, "压力")
    discrete_feature_list3 = feature_discretization(feature_list3, bucket_list2)

    feature_comb1 = [discrete_feature_list1, bucket_name_dict1]
    feature_comb2 = [discrete_feature_list2, bucket_name_dict2]
    feature_comb3 = [discrete_feature_list3, bucket_name_dict3]
    cross_feature_list1, index_name_dict1 = feature_cross_2(feature_comb1, feature_comb2, one_hot=False)
    cross_feature_list2, index_name_dict2 = feature_cross_3(feature_comb1, feature_comb2, feature_comb3, one_hot=False)

    feature_comb4 = [cross_feature_list1, index_name_dict1]
    feature_comb5 = [cross_feature_list2, index_name_dict2]
    feature_comb_list = [feature_comb1, feature_comb2, feature_comb3, feature_comb4, feature_comb5]
    sample_list, overall_bucket_name_dict = feature_concat(feature_comb_list)


def convert_dict(feature_name_dict):
    name = feature_name_dict[0]
    s = re.findall(r'[[](.*?)[)]', name)
    for v in s:
        name = name.replace(v, "")
    name = name.replace("[)", "").replace(" ", ", ")
    new_dict = {0: name}
    return new_dict


def eng2chi(name):
    name = name.replace("sum_play_day", "收听总天数").replace("sum_duration", "收听总时长")\
        .replace("collect_channel_cnt", "收藏专辑数").replace("program_cnt", "收听节目数")\
        .replace("collect_category_cnt", "收藏分类数").replace("chan_cnt", "收听专辑数")\
        .replace("category_cnt", "收听分类数").replace("share_channel_cnt", "分享数")\
        .replace("chat_cnt", "评论数")
    return name


