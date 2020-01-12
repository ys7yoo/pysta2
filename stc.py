import numpy as np
from sklearn.covariance import MinCovDet
from math import pi
from matplotlib import pyplot as plt

def flip_columns(vectors):
    column_sum = np.sum(vectors, axis=0)
    for i in range(len(column_sum)):
        if column_sum[i] < 0 :
            vectors[:,i] = -1 * vectors[:,i]
            
    return vectors

def sort_eigen_pair(eigen_values, eigen_vectors):
    sorted_eig_vectors = np.copy(eigen_vectors)
    sorted_eig_values = np.sort(eigen_values)[::-1]
    
    sorted_index = np.argsort(eigen_values)[::-1]
    original_index = list(range(0,len(eigen_values),1))
    
    sorted_eig_vectors[:,original_index] = eigen_vectors[:,sorted_index]

    return sorted_eig_values, sorted_eig_vectors


def calc_covariance_matrix(data_row, weights=None, centered=False):
    if centered:
        if weights is None:
            C = np.dot(data_row.T, data_row) / (data_row.shape[0]-1)
        else:
            data_row = np.dot(np.diag(np.sqrt(weights)), data_row)
            C = np.dot(data_row.T, data_row) / (np.sum(weights) - 1)
    else:
        C = np.cov(data_row, rowvar=False, fweights=weights)
    return 0.5*(C+C.T)


def calc_robust_covariance_matrix(data, centered=True):
    C = MinCovDet(random_state=0, assume_centered=centered).fit(data).covariance_
    return 0.5*(C+C.T)


def calc_eig_values_and_vectors(covariance_matrix):
    eig_values, eig_vectors = np.linalg.eigh(covariance_matrix)
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html

    eig_values, eig_vectors = sort_eigen_pair(eig_values, eig_vectors)

    return eig_values, force_point_positive(eig_vectors)


def force_point_positive(eig_vectors):
#     eig_vectors = eig_vectors.copy()
    idx_neg = np.sum(eig_vectors,axis=0) < 0
#     print(idx_neg)

    for i, val in enumerate(idx_neg):
        if val:
#             print(i)
            eig_vectors[:,i] = -eig_vectors[:,i]
    return eig_vectors


def project(data, eig_vectors):
    return np.dot(data, eig_vectors)

def calc_mahalanobis_distance(score, eigen_values):
    distance = list()
    inverse_covariance = np.diag(1 / eigen_values)
    
    for i in range(score.shape[0]):
        distance.append(np.sqrt(np.dot(np.dot(score[i], inverse_covariance), score[i].T)))
    return distance

def calc_robust_distance(data, score, eigen_vectors):
    perpendicular_foot = np.dot(score, eigen_vectors.T)
    diff = data - perpendicular_foot
 
    distance = np.sqrt(np.sum(diff**2,axis = 1))
    return distance

# 학생들 응답 평균치 높은 순서대로 다시 labeling
def sort_categories_by_mean(labels, sorted_index):
    sorted_categories_by_mean = np.zeros_like(labels)
    for i in range(len(sorted_index)):
        sorted_categories_by_mean[np.where(labels == sorted_index[i])[0]] = i + 1
    
    return sorted_categories_by_mean

# 그룹별로 요소에 대한 평균 구함
def calc_mean_elements_by_group(questions_per_element,group_mean, group_num):
    mean_per_element = dict()
    for element in list(questions_per_element.keys()):
        if len(questions_per_element[element]) == 1 :
            mean_per_element[element] = group_mean[group_num - 1][questions_per_element[element][0] - 1]
        else :
            a = list()
            for j in questions_per_element[element]:
                a.append(group_mean[group_num - 1][j - 1])
            mean_per_element[element] = np.mean(a)
            
    return mean_per_element

# radar graph 만들기
def make_spider(categories, values, row, ax, group):
  
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='grey', size=15)

    ax.set_rlabel_position(0)
    plt.yticks([-1,0,1], ["-1","0","1"], color="grey", size=12)
    plt.ylim(-2,2)


    values = list(values)
    values.append(values[0])
    #ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    #ax.fill(angles, values, color=color, alpha=0.4)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)

    plt.title(group, size=15, y=1.1)

    #my_palette = plt.cm.get_cmap("Set2", len(df.index))