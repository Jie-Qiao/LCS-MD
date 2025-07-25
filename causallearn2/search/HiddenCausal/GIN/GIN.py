'''
    File name: GIN.py
    Discription: Learning Hidden Causal Representation with GIN condition
    Author: ZhiyiHuang@DMIRLab, RuichuCai@DMIRLab
    From DMIRLab: https://dmir.gdut.edu.cn/
'''

import random
from collections import deque
from itertools import combinations

import numpy as np
from scipy.stats import chi2

from causallearn2.graph.GeneralGraph import GeneralGraph
from causallearn2.graph.GraphNode import GraphNode
from causallearn2.graph.NodeType import NodeType
from causallearn2.graph.Edge import Edge
from causallearn2.graph.Endpoint import Endpoint
from causallearn2.search.FCMBased.lingam.hsic import hsic_test_gamma
from causallearn2.utils.cit import kci


def fisher_test(pvals):
    pvals = [pval if pval >= 1e-5 else 1e-5 for pval in pvals]
    fisher_stat = -2.0 * np.sum(np.log(pvals))
    return 1 - chi2.cdf(fisher_stat, 2 * len(pvals))


def GIN(data, indep_test=kci, alpha=0.05):
    '''
    Learning causal structure of Latent Variables for Linear Non-Gaussian Latent Variable Model
    with Generalized Independent Noise Condition

    Parameters
    ----------
    data : numpy ndarray
           data set
    indep_test : callable, default=kci
        the function of the independence test being used
    alpha : float, default=0.05
        desired significance level of independence tests (p_value) in (0,1)
    Returns
    -------
    G : general graph
        causal graph
    K : list
        causal order
    '''
    n = data.shape[1]
    cov = np.cov(data.T)

    var_set = set(range(n))
    cluster_size = 2
    clusters_list = []
    while cluster_size < len(var_set):
        tmp_clusters_list = []
        for cluster in combinations(var_set, cluster_size):
            remain_var_set = var_set - set(cluster)
            e = cal_e_with_gin(data, cov, list(cluster), list(remain_var_set))
            pvals = []
            tmp_data = np.concatenate([data[:, list(remain_var_set)], e.reshape(-1, 1)], axis=1)
            for z in range(len(remain_var_set)):
                pvals.append(indep_test(tmp_data, z, - 1))
            fisher_pval = fisher_test(pvals)
            if fisher_pval >= alpha:
                tmp_clusters_list.append(cluster)
        tmp_clusters_list = merge_overlaping_cluster(tmp_clusters_list)
        clusters_list = clusters_list + tmp_clusters_list
        for cluster in tmp_clusters_list:
            var_set -= set(cluster)
        cluster_size += 1

    K = []
    updated = True
    while updated:
        updated = False
        X = []
        Z = []
        for cluster_k in K:
            cluster_k1, cluster_k2 = array_split(cluster_k, 2)
            X += cluster_k1
            Z += cluster_k2

        for i, cluster_i in enumerate(clusters_list):
            is_root = True
            random.shuffle(cluster_i)
            cluster_i1, cluster_i2 = array_split(cluster_i, 2)
            for j, cluster_j in enumerate(clusters_list):
                if i == j:
                    continue
                random.shuffle(cluster_j)
                cluster_j1, cluster_j2 = array_split(cluster_j, 2)
                e = cal_e_with_gin(data, cov, X + cluster_i1 + cluster_j1, Z + cluster_i2)
                pvals = []
                tmp_data = np.concatenate([data[:, Z + cluster_i2], e.reshape(-1, 1)], axis=1)
                for z in range(len(Z + cluster_i2)):
                    pvals.append(indep_test(tmp_data, z, - 1))
                fisher_pval = fisher_test(pvals)
                if fisher_pval < alpha:
                    is_root = False
                    break
            if is_root:
                K.append(cluster_i)
                clusters_list.remove(cluster_i)
                updated = True
                break

    G = GeneralGraph([])
    for var in var_set:
        o_node = GraphNode(f"X{var + 1}")
        G.add_node(o_node)

    latent_id = 1
    l_nodes = []

    for cluster in K:
        l_node = GraphNode(f"L{latent_id}")
        l_node.set_node_type(NodeType.LATENT)
        G.add_node(l_node)
        for l in l_nodes:
            G.add_directed_edge(l, l_node)
        l_nodes.append(l_node)

        for o in cluster:
            o_node = GraphNode(f"X{o + 1}")
            G.add_node(o_node)
            G.add_directed_edge(l_node, o_node)
        latent_id += 1

    undirected_l_nodes = []

    for cluster in clusters_list:
        l_node = GraphNode(f"L{latent_id}")
        l_node.set_node_type(NodeType.LATENT)
        G.add_node(l_node)
        for l in l_nodes:
            G.add_directed_edge(l, l_node)

        for l in undirected_l_nodes:
            G.add_edge(Edge(l, l_node, Endpoint.TAIL, Endpoint.TAIL))

        undirected_l_nodes.append(l_node)

        for o in cluster:
            o_node = GraphNode(f"X{o + 1}")
            G.add_node(o_node)
            G.add_directed_edge(l_node, o_node)
        latent_id += 1

    return G, K


def GIN_MI(data):
    '''
    Learning causal structure of Latent Variables for Linear Non-Gaussian Latent Variable Model
    with Generalized Independent Noise Condition

    Parameters
    ----------
    data : numpy ndarray
           data set

    Returns
    -------
    G : general graph
        causal graph
    K : list
        causal order
    '''
    v_labels = list(range(data.shape[1]))
    v_set = set(v_labels)
    cov = np.cov(data.T)

    # Step 1: Finding Causal Clusters
    cluster_list = []
    min_cluster = {i: set() for i in v_set}
    min_dep_score = {i: 1e9 for i in v_set}
    for (x1, x2) in combinations(v_set, 2):
        x_set = {x1, x2}
        z_set = v_set - x_set
        dep_statistic = cal_dep_for_gin(data, cov, list(x_set), list(z_set))
        for i in x_set:
            if min_dep_score[i] > dep_statistic:
                min_dep_score[i] = dep_statistic
                min_cluster[i] = x_set
    for i in v_labels:
        cluster_list.append(list(min_cluster[i]))

    cluster_list = merge_overlaping_cluster(cluster_list)

    # Step 2: Learning the Causal Order of Latent Variables
    K = []
    while (len(cluster_list) != 0):
        root = find_root(data, cov, cluster_list, K)
        K.append(root)
        cluster_list.remove(root)

    latent_id = 1
    l_nodes = []
    G = GeneralGraph([])
    for cluster in K:
        l_node = GraphNode(f"L{latent_id}")
        l_node.set_node_type(NodeType.LATENT)
        l_nodes.append(l_node)
        G.add_node(l_node)
        for l in l_nodes:
            if l != l_node:
                G.add_directed_edge(l, l_node)
        for o in cluster:
            o_node = GraphNode(f"X{o + 1}")
            G.add_node(o_node)
            G.add_directed_edge(l_node, o_node)
        latent_id += 1

    return G, K


def cal_e_with_gin(data, cov, X, Z):
    cov_m = cov[np.ix_(Z, X)]
    _, _, v = np.linalg.svd(cov_m)
    omega = v.T[:, -1]
    return np.dot(omega, data[:, X].T)


def cal_dep_for_gin(data, cov, X, Z):
    '''
    Calculate the statistics of dependence via Generalized Independent Noise Condition

    Parameters
    ----------
    data : data set (numpy ndarray)
    cov : covariance matrix
    X : test set variables
    Z : condition set variables

    Returns
    -------
    sta : test statistic
    '''

    e_xz = cal_e_with_gin(data, cov, X, Z)

    sta = 0
    for i in Z:
        sta += hsic_test_gamma(e_xz, data[:, i])[0]
    sta /= len(Z)
    return sta


def find_root(data, cov, clusters, K):
    '''
    Find the causal order by statistics of dependence

    Parameters
    ----------
    data : data set (numpy ndarray)
    cov : covariance matrix
    clusters : clusters of observed variables
    K : causal order

    Returns
    -------
    root : latent root cause
    '''
    if len(clusters) == 1:
        return clusters[0]
    root = clusters[0]
    dep_statistic_score = 1e30
    for i in clusters:
        for j in clusters:
            if i == j:
                continue
            X = [i[0], j[0]]
            Z = []
            for k in range(1, len(i)):
                Z.append(i[k])

            if K:
                for k in K:
                    X.append(k[0])
                    Z.append(k[1])

            dep_statistic = cal_dep_for_gin(data, cov, X, Z)
            if dep_statistic < dep_statistic_score:
                dep_statistic_score = dep_statistic
                root = i

    return root


def _get_all_elements(S):
    result = set()
    for i in S:
        for j in i:
            result |= {j}
    return result


# merging cluster
def merge_overlaping_cluster(cluster_list):
    v_labels = _get_all_elements(cluster_list)
    if len(v_labels) == 0:
        return []
    cluster_dict = {i: -1 for i in v_labels}
    cluster_b = {i: [] for i in v_labels}
    cluster_len = 0
    for i in range(len(cluster_list)):
        for j in cluster_list[i]:
            cluster_b[j].append(i)

    visited = [False] * len(cluster_list)
    cont = True
    while cont:
        cont = False
        q = deque()
        for i, val in enumerate(visited):
            if not val:
                q.append(i)
                visited[i] = True
                break
        while q:
            top = q.popleft()
            for i in cluster_list[top]:
                cluster_dict[i] = cluster_len
                for j in cluster_b[i]:
                    if not visited[j]:
                        q.append(j)
                        visited[j] = True

        for i in visited:
            if not i:
                cont = True
                break
        cluster_len += 1

    cluster = [[] for _ in range(cluster_len)]
    for i in v_labels:
        cluster[cluster_dict[i]].append(i)

    return cluster


def array_split(x, k):
    x_len = len(x)
    # div_points = []
    sub_arys = []
    start = 0
    section_len = x_len // k
    extra = x_len % k
    for i in range(extra):
        sub_arys.append(x[start:start + section_len + 1])
        start = start + section_len + 1

    for i in range(k - extra):
        sub_arys.append(x[start:start + section_len])
        start = start + section_len
    return sub_arys
