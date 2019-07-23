import numpy as np
import pandas as pd
import collections
from preprocessing import *
from feature_selection import *


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
    label_list = last_flag
    # 分桶
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x!=0, program_cnt)), bucket_num=20)
    # program_cnt_buckets = [1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 19, 24, 32, 43, 61, 93, 162]
    program_cnt_buckets = [i for i in range(0, 162, 3)]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, chan_cnt)), bucket_num=15)
    # chan_cnt_buckets = [1, 2, 3, 4, 5, 7, 9, 13]
    chan_cnt_buckets = [i for i in range(20)]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, category_cnt)), bucket_num=10)
    # category_cnt_buckets = [1, 2, 3, 4]
    category_cnt_buckets = [i for i in range(10)]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, sum_duration)), bucket_num=20)
    # sum_duration_buckets = [23, 62, 138, 289, 572, 1024, 1630, 2457, 3569, 5095, 7208, 10200, 14376, 20240, 28447, 41039, 61738, 100081, 185248]
    sum_duration_buckets = [i for i in range(0, 100000, 3000)]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, sum_play_day)), bucket_num=15)
    # sum_play_day_buckets = [1, 2, 3, 4, 5, 6, 8, 10, 13]
    sum_play_day_buckets = [i for i in range(20)]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, collect_channel_cnt)), bucket_num=15)
    # collect_channel_cnt_buckets = [1, 2, 3, 4, 9]
    collect_channel_cnt_buckets = [i for i in range(20)]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, collect_category_cnt)), bucket_num=10)
    # collect_category_cnt_buckets = [1, 2, 3, 4, 7, 13]
    collect_category_cnt_buckets = [i for i in range(20)]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, chat_cnt)), bucket_num=40)
    # chat_cnt_buckets = [1, 2]
    chat_cnt_buckets = [i for i in range(10)]
    # bucket_list = even_num_bucketing(feature_list=list(filter(lambda x: x != 0, share_channel_cnt)), bucket_num=40)
    # share_channel_cnt_buckets = [1, 2, 3]
    share_channel_cnt_buckets = [i for i in range(10)]

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

    program_cnt_comb = [program_cnt_features, program_cnt_bucket_name]
    chan_cnt_comb = [chan_cnt_features, chan_cnt_bucket_name]
    category_cnt_comb = [category_cnt_features, category_cnt_bucket_name]
    sum_duration_comb = [sum_duration_features, sum_duration_bucket_name]
    sum_play_day_comb = [sum_play_day_features, sum_play_day_bucket_name]
    collect_channel_cnt_comb = [collect_channel_cnt_features, collect_channel_cnt_bucket_name]
    collect_category_cnt_comb = [collect_category_cnt_features, collect_category_cnt_bucket_name]
    chat_cnt_comb = [chat_cnt_features, chat_cnt_bucket_name]
    share_channel_cnt_comb = [share_channel_cnt_features, share_channel_cnt_bucket_name]

    # 一阶留存率曲线
    draw_retention_rate_2d(sum_play_day_comb, label_list, "Sum Play Day")
    draw_retention_rate_2d(sum_duration_comb, label_list, "Sum Duration")
    draw_retention_rate_2d(program_cnt_comb, label_list, "Program Count")
    draw_retention_rate_2d(category_cnt_comb, label_list, "Category Count")
    draw_retention_rate_2d(chan_cnt_comb, label_list, "Channel Count")
    draw_retention_rate_2d(collect_channel_cnt_comb, label_list, "Collect Channel Count")
    draw_retention_rate_2d(collect_category_cnt_comb, label_list, "Collect Category Count")
    draw_retention_rate_2d(share_channel_cnt_comb, label_list, "Share Channel Count")
    draw_retention_rate_2d(chat_cnt_comb, label_list, "Chat Count")

    # 二阶阶留存率图像
    draw_retention_rate_3d(program_cnt_comb, chan_cnt_comb, label_list, "chan_cnt * program_cnt/3")
    # draw_retention_rate_3d(collect_channel_cnt_comb, chat_cnt_comb, label_list, "chat_cnt * collect_channel_cnt")
    draw_retention_rate_3d(collect_channel_cnt_comb, collect_category_cnt_comb, label_list, "collect_category_cnt * collect_channel_cnt")
    draw_retention_rate_3d(sum_duration_comb, collect_channel_cnt_comb, label_list, "collect_channel_cnt * sum_duration/3000")
    draw_retention_rate_3d(program_cnt_comb, category_cnt_comb, label_list, "category_cnt * program_cnt/3")
    # draw_retention_rate_3d(sum_duration_comb, sum_play_day_comb, label_list, "sum_play_day * sum_duration/3000")
    draw_retention_rate_3d(chan_cnt_comb, collect_channel_cnt_comb, label_list, "collect_channel_cnt * chan_cnt")
    draw_retention_rate_3d(program_cnt_comb, sum_duration_comb, label_list, "sum_duration/3000 * program_cnt/3")
    draw_retention_rate_3d(sum_duration_comb, collect_category_cnt_comb, label_list, "collect_category_cnt * sum_duration/3000")
    draw_retention_rate_3d(chan_cnt_comb, sum_duration_comb, label_list, "sum_duration/3000 * chan_cnt")
    # draw_retention_rate_3d(program_cnt_comb, sum_play_day_comb, label_list, "sum_play_day * program_cnt/3")
