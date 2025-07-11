import pandas
import math
import numpy as np
import random
from scipy.special import expit as sigmoid

from itertools import product
from lib.utils import add_missing
import networkx as nx
def randomGraph(dim=10, indegree=1,randomState=None):
    if randomState is None:
        randomState=np.random.RandomState()
    edge_mat = np.zeros([dim, dim])
    edge_select = list(filter(lambda i: i[0] < i[1], product(range(dim), range(dim))))
    randomState.shuffle(edge_select)
    for edge_ind in edge_select[:round(indegree * dim)]:
        edge_mat[edge_ind] = 1
    return edge_mat



def linear_data(B_true, sample_size=5000, sem_type="gauss", w_ranges=((-3.0, -1.0), (1.0, 3.0)), randomState=None,p=0.05):
    if randomState is None:
        randomState=np.random.RandomState()
    W_true = simulate_parameter(B_true,w_ranges=w_ranges,randomState=randomState)
    #print(W_true)
    data = simulate_linear_sem(W_true, sample_size, sem_type, randomState=randomState,p= p)
    # Nan_num = np.isnan(data).sum()
    # print(Nan_num)
    col=[str(i) for i in range(len(B_true))]
    data = pandas.DataFrame(data,columns=col)
    return data

def nonlinear_data(B_true, sample_size=5000, sem_type="mim", randomState=None):
    if randomState is None:
        randomState=np.random.RandomState()

    data = simulate_nonlinear_sem(B_true, sample_size, sem_type =sem_type, randomState=randomState)
    # print(data)
    # Nan_num = np.isnan(data).sum()
    # print(Nan_num)
    col=[str(i) for i in range(len(B_true))]
    data = pandas.DataFrame(data,columns=col)
    return data

def simulate_nonlinear_sem(B, sample_size, sem_type ='mim', noise_scale=None, randomState=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        sample_size (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive missingness, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    if randomState is None:
        randomState=np.random.RandomState()

    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        #z = randomState.normal(scale=scale, size=sample_size)
        z = randomState.uniform(low=-scale, high=scale, size=sample_size)
        # z = power_with_original_sign(z, 0.5)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 50
            W1 = randomState.uniform(low=0.5, high=1.0, size=[pa_size, hidden])
            W1[randomState.rand(*W1.shape) < 0.5] *= -1                       # * 列表变量 ：将列表元素分开，
            W2 = randomState.uniform(low=0.5, high=1.0, size=hidden)
            W2[randomState.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
            #x = np.tanh(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = randomState.uniform(low=0, high=1, size=pa_size)
            w1[randomState.rand(pa_size) < 0.5] *= -1
            w2 = randomState.uniform(low=0, high=1, size=pa_size)
            w2[randomState.rand(pa_size) < 0.5] *= -1
            w3 = randomState.uniform(low=0, high=1, size=pa_size)
            w3[randomState.rand(pa_size) < 0.5] *= -1
            W4 = randomState.uniform(low=0.5, high=1, size=pa_size)
            W4[randomState.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + X @ W4 + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]    # 获取节点数
    scale_vec = noise_scale if noise_scale else np.ones(d)  #None和FALSE 选择np.ones(d) ; True选择noise_scale
    X = np.zeros([sample_size, d])
    #nx.from_numpy_array()
    G=nx.from_numpy_array(B,create_using=nx.DiGraph)
    ordered_vertices=list(nx.topological_sort(G))
    # G = ig.Graph.Adjacency(B.tolist())
    # ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
        X[:, j]= (X[:, j]-np.mean(X[:, j]))/np.std(X[:, j])
    return X

def test_data(randomState=None):
    if randomState is None:
        randomState=np.random.RandomState()
    sz = 5000
    data = np.zeros((sz, 7))

    p =  0.5
    X = power_with_original_sign(randomState.normal(0, 1.0, size=sz), p)
    Z = 2 * X + power_with_original_sign(randomState.normal(0, 1.0, size=sz), p)

    Y = -1* Z + power_with_original_sign(randomState.normal(0, 1.0, size=sz), p)
    W = 1.5 * X + 1.8 * Y + power_with_original_sign(randomState.normal(0, 1.0, size=sz), p)

    U = power_with_original_sign(randomState.normal(0, 1.0, size=sz), p)
    O = 3 * U + power_with_original_sign(randomState.normal(0, 1.0, size=sz), p)
    P = 1.9 * O + power_with_original_sign(randomState.normal(0, 1.0, size=sz), p)

    data[:, 0], data[:, 1], data[:, 2], data[:, 3] = X, Y, Z, W

    data[:, 4] = U
    data[:, 5] = O
    data[:, 6] = P

    mdata = data.copy()

    # X--> Z -->Y
    # X--> W <--Y
    # X--> W2<--Y
    # W --> Rx
    # W2 --> Ry
    # U --> Rw2
    str_3 = [str(3)]
    str_2 = [str(2)]
    str_5 = [str(5)]

    col = [str(i) for i in range(data.shape[1])]
    data = pandas.DataFrame(data, columns=col)

    cause_dict = {'1': str_3, '2': str_2, '5': str_5}
    # data = pd.DataFrame(data)
    mdata = add_missing(data, cause_dict=cause_dict, m_min=0.1, m_max=0.8, quantile=0.2, randomState=randomState)

    # mdata[W > 0, 1] = np.nan
    # mdata[O > 0, 5] = np.nan
    # mdata[Z > 0, 2] = np.nan
    return mdata, cause_dict

def simulate_parameter(B, w_ranges=((-3.0, -1.0), (1.0, 3.0)),randomState=None):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    if randomState is None:
        randomState=np.random.RandomState()
    W = np.zeros(B.shape)
    S = randomState.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = randomState.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U           # S == i 如何理解？ 它的作用类似过滤器，i看成全0或者全1的矩阵，S == i后会形成一个布尔矩阵，只有元素为true的随机权重才可以保留下来
        # print(S == i)
    return W

def simulate_linear_sem(W, n, sem_type, noise_scale=None,randomState=None, p= 0.05):
    """Simulate samples from linear SEM with specified type of missingness.

    For uniform, missingness z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive missingness, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    if randomState is None:
        randomState=np.random.RandomState()

    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = randomState.normal(scale=scale, size=n)               # 噪声从标准差为scale的高斯分布中取样

            x = X @ w + z                                           # 通过parents的取样和权重系数相乘加噪声生成 n * 1 维的矩阵，即某个变量的采样
        elif sem_type == 'exp':                                     # 外生变量本身就是一种噪声
            z = randomState.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'sub-gauss':
            z = randomState.normal(scale=scale, size=n)  # 噪声从标准差为scale的高斯分布中取样
            z = power_with_original_sign(z, p)  # p 指数
            # Nan_num = np.isnan(z).sum()
            # print(Nan_num)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = randomState.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = randomState.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = randomState.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = randomState.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('missingness scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G=nx.from_numpy_array(W,create_using=nx.DiGraph)
    ordered_vertices=list(nx.topological_sort(G))         # 找出图的拓扑序列
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])

    # scale_vec[0] = pow(2, 1)
    # scale_vec[1] = pow(2, 2)
    # scale_vec[2] = pow(2, 0)
    # scale_vec[3] = pow(2, 1)
    # scale_vec[4] = pow(2, 0)
    # scale_vec[5] = pow(2, 3)
    # scale_vec[6] = pow(2, 3)
    for j in ordered_vertices:
        # scale_vec[j] = pow(2, ordered_vertices.index(j))
        parents = list(G.predecessors(j))           # 找到当前结点的父亲结点
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X


def power_with_original_sign(num_array, pow_num):
    size = num_array.shape[0]
    result = np.zeros(size)
    for i in range(size):
        if num_array[i] >= 0:
            result[i] = math.pow(num_array[i], pow_num)
        else:
            result[i] = - math.pow(abs(num_array[i]), pow_num)
    return result

def is_dag(W):
    G = nx.from_numpy_array(W, create_using=nx.DiGraph)
    return nx.is_directed_acyclic_graph(G)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)