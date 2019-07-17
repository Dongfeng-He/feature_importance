from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from mpl_toolkits.mplot3d import Axes3D # 虽然不直接调用，但不能删除
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm


def draw_hist(heights, interval_num, xlabel="", ylabel="", title=""):
    # 创建直方图
    # 第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
    # 第二个参数为划分的区间个数
    plt.hist(heights, interval_num)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def draw_line(x, y, name_list):
    x, y = (list(t) for t in zip(*sorted(zip(x, y), reverse=False)))
    plt.plot(x, y)
    plt.xticks(x[::1], name_list[::1], rotation=45)
    plt.show()


def draw_3d(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #  Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens
    # OrRd, CMRmap_r
    cm = plt.cm.get_cmap('OrRd')
    color_num = 10
    color_list = []
    interval = (max(z) - min(z)) / color_num
    for i in range(len(z)):
        for j in range(1, color_num + 1):
            if z[i] < j * interval:
                break
        color_list.append(cm(float(j) / color_num * 0.6 + 0.2))
    # Grab some test data.
    # x, y, z = axes3d.get_test_data(0.05)
    # Plot a basic wireframe.
    #ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
    #surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.scatter(x, y, z, c=color_list)
    plt.show()


def draw_3d_2(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data.
    x, y = np.meshgrid(x, y)
    z = np.array(z).transpose()
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def draw_retention_rate_2d(feature_comb, label_list):
    feature_list, bucket_name_dict = feature_comb[0], feature_comb[1]
    feature_size = max(feature_list) + 1
    feature_retention_dict = {feature: [0, 0] for feature in range(feature_size)}
    for feature, label in zip(feature_list, label_list):
        feature_retention_dict[feature][label] += 1
    feature_retention_rate_dict = {}
    x = []
    y = []
    name_list = []
    for feature in range(feature_size):
        positive_num = feature_retention_dict[feature][1]
        total_num = sum(feature_retention_dict[feature])
        retention_rate = positive_num / total_num
        feature_retention_rate_dict[feature] = retention_rate
        x.append(feature)
        y.append(retention_rate)
        name_list.append(bucket_name_dict[feature])
    draw_line(x, y, name_list)
    print()


def draw_retention_rate_3d(feature_comb1, feature_comb2, label_list):
    feature_list1, bucket_name_dict1 = feature_comb1[0], feature_comb1[1]
    feature_size1 = max(feature_list1) + 1
    feature_list2, bucket_name_dict2 = feature_comb2[0], feature_comb2[1]
    feature_size2 = max(feature_list2) + 1

    retention_num_matrix = np.zeros([feature_size1, feature_size2])
    retention_rate_matrix = np.zeros([feature_size1, feature_size2])
    total_num_matrix = np.zeros([feature_size1, feature_size2])
    for feature1, feature2, label in zip(feature_list1, feature_list2, label_list):
        total_num_matrix[feature1][feature2] += 1
        if label == 1:
            retention_num_matrix[feature1][feature2] += 1
    for i in range(feature_size1):
        for j in range(feature_size2):
            if total_num_matrix[i][j] != 0:
                retention_rate_matrix[i][j] = retention_num_matrix[i][j] / total_num_matrix[i][j]
    x = [i for i in range(feature_size1)]
    y = [i for i in range(feature_size2)]
    z = retention_rate_matrix
    draw_3d_2(x, y, z)
    print()


def multivariate_pearsonr(X, y):
    scores, pvalues = [], []
    for ret in map(lambda x: pearsonr(x, y), X.T):
        scores.append(abs(ret[0]))
        pvalues.append(ret[1])
    return np.array(scores), np.array(pvalues)


class LR(LogisticRegression):
    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        #权值相近的阈值
        self.threshold = threshold
        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                 fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                 random_state=random_state, solver=solver, max_iter=max_iter,
                 multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)
        #使用同样的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        #训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        #训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        #权值系数矩阵的行数对应目标值的种类数目
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                #L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    #对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        #在L2逻辑回归中，权值系数之差小于设定的阈值，且在L1中对应的权值为0
                        if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    #计算这一类特征的权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self


if __name__ == "__main__":
    iris = load_iris()
    x = iris.data[:, 2]
    y = iris.data[:, 3]
    z = iris.target
    draw_3d(x, y, z)
    # draw_line(x, z)

    # 皮尔逊系数
    # 协方差 / 标准差
    # cov = E((X - EX)(Y - EY))，当 X 高于它的期望，Y 随之高于它的期望，两者乘积就打，变化趋势一样，协方差就大，相反，协方差就小
    # features1 = SelectKBest(score_func=multivariate_pearsonr, k=2).fit_transform(iris.data, iris.target)
    feature_pearsonr = multivariate_pearsonr(iris.data, iris.target)[0]
    # 卡方检验（对自变量每一个值都是当成离散值对待）
    # 对于一列特征，全部混在一起，每一种取值的数量一个期望。把特征按照不同的输出类别分组，每组里面每一种取值有一个数量，这个数量与这个取值在这个组（这么多样本数情况下）的数量期望，差距越大越好
    # features2 = SelectKBest(score_func=chi2, k=2).fit_transform(iris.data, iris.target)
    feature_chi = chi2(iris.data, iris.target)[0]
    # 互信息（感觉也是像离散值，自变量与类别的共现概率）
    # features3 = SelectKBest(score_func=mutual_info_classif, k=2).fit_transform(iris.data, iris.target)
    feature_mutual_info = mutual_info_classif(iris.data, iris.target)

    gbdt = GradientBoostingClassifier()
    gbdt_model = gbdt.fit(X=iris.data, y=iris.target)
    gbdt_feat_importance = gbdt_model.feature_importances_

    # 用 l1 惩罚的时候，归为0不代表一定不重要，而是对目标值有相同相关性的特征只留下一个，比如2、3特征与y相关性很强，但只保留了2，3变为0
    # 但是这个能得到最重要的特征吗
    lr = LogisticRegression(penalty="l2", C=0.1)
    lr_model = lr.fit(X=iris.data, y=iris.target)

    lr_mix = LR(threshold=0.5, C=0.1)
    lr_mix_model = lr_mix.fit(X=iris.data, y=iris.target)
    print()




