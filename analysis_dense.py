import collections
import numpy as np
import pandas as pd
from preprocessing import *
from feature_selection import *
import xgboost
import random
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import os
import time
import itertools


if __name__ == "__main__":
    csv_path = "magic_number_feature.csv"
    df = pd.read_csv(csv_path)
    # 是否留存
    last_flag = df["last_flag"].values.tolist()
    # 收听节目个数
    program_cnt = df["program_cnt"].values.tolist()
    # 收听专辑个数
    chan_cnt = df["chan_cnt"].values.tolist()
    # 收听分类个数
    category_cnt = df["category_cnt"].values.tolist()
    # 累计收听时长
    sum_duration = df["sum_duration"].values.tolist()
    # 累计收听天数
    sum_play_day = df["sum_play_day"].values.tolist()
    # 收藏专辑个数
    collect_channel_cnt = df["collect_channel_cnt"].values.tolist()
    # 收藏分类个数
    collect_category_cnt = df["collect_category_cnt"].values.tolist()
    # 发表评论数量
    chat_cnt = df["chat_cnt"].values.tolist()
    # 分享数
    share_channel_cnt = df["share_channel_cnt"].values.tolist()
    # 签到个数（计算的是qingting_id，会大于15，不用）
    signin_cnt = df["signin_cnt"].values.tolist()

    # 统计留存比例（0: 853553, 1: 180340）
    last_counter = collections.Counter(last_flag)
    chat_cnt_counter = collections.Counter(chat_cnt)
    share_channel_cnt_counter = collections.Counter(share_channel_cnt)
    program_cnt_counter = collections.Counter(program_cnt)
    chan_cnt_counter = collections.Counter(chan_cnt)
    sum_duration_counter = collections.Counter(sum_duration)
    label_list = last_flag

    # 样本均衡
    sample_balance = True
    if sample_balance:
        sample_pairs = list(zip(label_list, program_cnt, chan_cnt, category_cnt, sum_duration, sum_play_day, collect_channel_cnt, collect_category_cnt, chat_cnt, share_channel_cnt))
        positive_pairs = list(filter(lambda x: x[0] == 1, sample_pairs))
        negative_pairs = list(filter(lambda x: x[0] == 0, sample_pairs))
        negative_pairs_balanced = random.sample(negative_pairs, len(positive_pairs))
        sample_pairs = positive_pairs + negative_pairs_balanced
        random.shuffle(sample_pairs)
        # sample_pairs = random.sample(sample_pairs, 1000)
        label_list = list(map(lambda x: x[0], sample_pairs))
        program_cnt = list(map(lambda x: x[1], sample_pairs))
        chan_cnt = list(map(lambda x: x[2], sample_pairs))
        category_cnt = list(map(lambda x: x[3], sample_pairs))
        sum_duration = list(map(lambda x: x[4], sample_pairs))
        sum_play_day = list(map(lambda x: x[5], sample_pairs))
        collect_channel_cnt = list(map(lambda x: x[6], sample_pairs))
        collect_category_cnt = list(map(lambda x: x[7], sample_pairs))
        chat_cnt = list(map(lambda x: x[8], sample_pairs))
        share_channel_cnt = list(map(lambda x: x[9], sample_pairs))

    # 分桶
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x!=0, program_cnt)), bucket_num=20)
    program_cnt_buckets = [1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 19, 24, 32, 43, 61, 93, 162]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, chan_cnt)), bucket_num=15)
    chan_cnt_buckets = [1, 2, 3, 4, 5, 7, 9, 13]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, category_cnt)), bucket_num=10)
    category_cnt_buckets = [1, 2, 3, 4]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, sum_duration)), bucket_num=20)
    sum_duration_buckets = [23, 62, 138, 289, 572, 1024, 1630, 2457, 3569, 5095, 7208, 10200, 14376, 20240, 28447, 41039, 61738, 100081, 185248]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, sum_play_day)), bucket_num=15)
    sum_play_day_buckets = [1, 2, 3, 4, 5, 6, 8, 10, 13]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, collect_channel_cnt)), bucket_num=15)
    collect_channel_cnt_buckets = [1, 2, 3, 4, 9]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, collect_category_cnt)), bucket_num=10)
    collect_category_cnt_buckets = [1, 2, 3, 4, 7, 13]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, chat_cnt)), bucket_num=40)
    chat_cnt_buckets = [1, 2]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, share_channel_cnt)), bucket_num=40)
    share_channel_cnt_buckets = [1, 2, 3]

    # 离散化
    program_cnt_bucket_name = create_bucket_name_dict(program_cnt_buckets, "program_cnt")
    program_cnt_features = feature_discretization(program_cnt, program_cnt_buckets)
    chan_cnt_bucket_name = create_bucket_name_dict(chan_cnt_buckets, "chan_cnt")
    chan_cnt_features = feature_discretization(chan_cnt, chan_cnt_buckets)
    category_cnt_bucket_name = create_bucket_name_dict(category_cnt_buckets, "category_cnt")
    category_cnt_features = feature_discretization(category_cnt, category_cnt_buckets)
    sum_duration_bucket_name = create_bucket_name_dict(sum_duration_buckets, "sum_duration")
    sum_duration_features = feature_discretization(sum_duration, sum_duration_buckets)
    sum_play_day_bucket_name = create_bucket_name_dict(sum_play_day_buckets, "sum_play_day")
    sum_play_day_features = feature_discretization(sum_play_day, sum_play_day_buckets)
    collect_channel_cnt_bucket_name = create_bucket_name_dict(collect_channel_cnt_buckets, "collect_channel_cnt")
    collect_channel_cnt_features = feature_discretization(collect_channel_cnt, collect_channel_cnt_buckets)
    collect_category_cnt_bucket_name = create_bucket_name_dict(collect_category_cnt_buckets, "collect_category_cnt")
    collect_category_cnt_features = feature_discretization(collect_category_cnt, collect_category_cnt_buckets)
    chat_cnt_bucket_name = create_bucket_name_dict(chat_cnt_buckets, "chat_cnt")
    chat_cnt_features = feature_discretization(chat_cnt, chat_cnt_buckets)
    share_channel_cnt_bucket_name = create_bucket_name_dict(share_channel_cnt_buckets, "share_channel_cnt")
    share_channel_cnt_features = feature_discretization(share_channel_cnt, share_channel_cnt_buckets)

    feature_comb_0 = [program_cnt_features, program_cnt_bucket_name]
    feature_comb_1 = [chan_cnt_features, chan_cnt_bucket_name]
    feature_comb_2 = [category_cnt_features, category_cnt_bucket_name]
    feature_comb_3 = [sum_duration_features, sum_duration_bucket_name]
    feature_comb_4 = [sum_play_day_features, sum_play_day_bucket_name]
    feature_comb_5 = [collect_channel_cnt_features, collect_channel_cnt_bucket_name]
    feature_comb_6 = [collect_category_cnt_features, collect_category_cnt_bucket_name]
    feature_comb_7 = [chat_cnt_features, chat_cnt_bucket_name]
    feature_comb_8 = [share_channel_cnt_features, share_channel_cnt_bucket_name]
    """
    feature_comb_list = [feature_comb_0, feature_comb_1, feature_comb_2, feature_comb_3, feature_comb_4,
                         feature_comb_5, feature_comb_6, feature_comb_7, feature_comb_8]
    """
    feature_comb_list = [feature_comb_1, feature_comb_2, feature_comb_5, feature_comb_6]

    dense_feature_comb_0 = [program_cnt, convert_dict(program_cnt_bucket_name)]
    dense_feature_comb_1 = [chan_cnt, convert_dict(chan_cnt_bucket_name)]
    dense_feature_comb_2 = [category_cnt, convert_dict(category_cnt_bucket_name)]
    dense_feature_comb_3 = [sum_duration, convert_dict(sum_duration_bucket_name)]
    dense_feature_comb_4 = [sum_play_day, convert_dict(sum_play_day_bucket_name)]
    dense_feature_comb_5 = [collect_channel_cnt, convert_dict(collect_channel_cnt_bucket_name)]
    dense_feature_comb_6 = [collect_category_cnt, convert_dict(collect_category_cnt_bucket_name)]
    dense_feature_comb_7 = [chat_cnt, convert_dict(chat_cnt_bucket_name)]
    dense_feature_comb_8 = [share_channel_cnt, convert_dict(share_channel_cnt_bucket_name)]
    """
    dense_feature_comb_list = [dense_feature_comb_0, dense_feature_comb_1, dense_feature_comb_2, dense_feature_comb_3,
                               dense_feature_comb_4, dense_feature_comb_5, dense_feature_comb_6, dense_feature_comb_7,
                               dense_feature_comb_8]
    """
    dense_feature_comb_list = [dense_feature_comb_1, dense_feature_comb_2, dense_feature_comb_5, dense_feature_comb_6]
    if True:
        # 特征交叉, 二阶
        feature_cate_num = len(feature_comb_list)
        for i in range(0, feature_cate_num - 1):
            for j in range(i + 1, feature_cate_num):
                cross_feature_list, cross_feature_name_dict = feature_cross_2(feature_comb_list[i], feature_comb_list[j], one_hot=False)
                new_dict = convert_dict(cross_feature_name_dict)
                feature_comb = [cross_feature_list, new_dict]
                dense_feature_comb_list.append(feature_comb)

    if False:
        # 特征交叉, 三阶
        for combs in itertools.combinations(feature_comb_list, 3):
            cross_feature_list, cross_feature_name_dict = feature_cross_3(combs[0], combs[1], combs[2], one_hot=False)
            new_dict = convert_dict(cross_feature_name_dict)
            feature_comb = [cross_feature_list, new_dict]
            dense_feature_comb_list.append(feature_comb)

    # 训练 XGBoost
    sample_list, overall_bucket_name_dict = feature_concat_dense(dense_feature_comb_list)
    overall_bucket_name_list = [overall_bucket_name_dict[i] for i in range(len(overall_bucket_name_dict))]
    # 训练 XGBoost
    sample_rate = 1
    split_rate = 0.8
    sample_num = int(len(sample_list) * sample_rate)
    sample_list = sample_list[: sample_num]
    label_list = label_list[: sample_num]
    train_num = int(len(sample_list) * split_rate)
    random.seed(1)
    random.shuffle(sample_list)
    random.seed(1)
    random.shuffle(label_list)
    x_valid = np.array(sample_list[train_num:])
    y_valid = np.array(label_list[train_num:])
    x_train = np.array(sample_list[:train_num])
    y_train = np.array(label_list[:train_num])

    # 参数搜索
    if False:
        # 一阶: [0.1, 292, 3, 10, 0.49370939396193736, 0.7, 0.6, 1, 3]
        # 二阶: [0.05, 158, 4, 7, 0.5567969076614763, 0.8, 0.9, 0.05, 2]
        # 三阶: [0.05, 437, 6, 1, 0.28736142018256616, 0.9, 0.9, 3, 2]
        grid_search(x_train, y_train, x_valid, y_valid)

    # 初始化 XGBoost 参数
    if True:
        [learning_rate, n_estimators, max_depth, min_child_weight, gamma, subsample, colsample_bytree, reg_alpha,
         reg_lambda] = [0.01, 500, 7, 2, 0.22, 0.7, 0.7, 1, 0.05]
        # 使用 GPU 加速
        if os.path.exists("/root/feature_importance"):
            classifier = xgboost.XGBClassifier(n_jobs=-1, random_state=0, seed=10, learning_rate=learning_rate,
                                               n_estimators=n_estimators, max_depth=max_depth,
                                               min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,
                                               colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                                               reg_lambda=reg_lambda, tree_method='gpu_hist')
        # 不使用 GPU 加速
        else:
            classifier = xgboost.XGBClassifier(n_jobs=-1, random_state=0, seed=10, learning_rate=learning_rate,
                                               n_estimators=n_estimators, max_depth=max_depth,
                                               min_child_weight=min_child_weight, gamma=gamma, subsample=subsample,
                                               colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                                               reg_lambda=reg_lambda)

    # 开始训练
    start_time = time.time()
    classifier.fit(x_train, y_train)
    print("耗时：%d min" % int((time.time() - start_time) / 60))
    y_pred = classifier.predict(x_valid)
    result = precision_recall_fscore_support(y_valid, y_pred)
    auc_score = roc_auc_score(y_valid, y_pred)
    print("XGBoost分类器：「准确率:{:.2%}」 「召回率:{:.2%}」 「F1_score:{:.2%}」 「AUC_score:{:.2%}」".format(float(result[0][1]), float(result[1][1]), float(result[2][1]), auc_score))
    feature_importance_pairs = list(zip(overall_bucket_name_list, classifier.feature_importances_))
    sorted_feature_importance = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)
    # 打印特征重要性
    for i in range(len(sorted_feature_importance)):
        print("%d\t%s\t%f" % (i + 1, eng2chi(sorted_feature_importance[i][0]), sorted_feature_importance[i][1]))

