from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = "1"

from itertools import combinations
from warnings import warn
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from copy import deepcopy
import pandas as pd
import xgboost
from causallearn2.graph.GraphClass import CausalGraph
from causallearn2.utils.cit import CIT
from causallearn2.utils.PCUtils import Helper
from causallearn2.graph.Edge import Edge
from causallearn2.graph.Endpoint import Endpoint

from sklearn.linear_model import LinearRegression
# from castle.common.independence_tests import hsic_test
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
from independence_testing.HSICBlockTestObject import HSICBlockTestObject
from kerpy.GaussianKernel import GaussianKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from lib.ANM_rule_path import is_cycle,parent
from lib.ANM_rule_path import find_R_descendent_paths,is_bi_edge
# from castle.algorithms import ANMNonlinear
from numba.typed import List
from causallearn2.utils.KCI.KCI import KCI_UInd


def filter_related_path(G,paths,all_bi_edges,path_id=-1):
    # only reserve the path that could affect the identifiable of anm.
    n = len(G) // 2
    if path_id==-1:
        idx=range(n)
    else:
        idx = range(path_id,path_id+1)
    for i in idx:
        pa_i=parent(G,i)
        pop_list=[]
        for j,p in enumerate(paths[i]):
            affect_identifiability=False
            for x in pa_i:
                if (x,i) not in all_bi_edges and (i,x) not in all_bi_edges:
                    continue
                # 只要任意一个x作为parent使得不可识别，都是需要保留的
                # 如果对于所有x作为parent都不会导致不可识别，则可删除
                # x 不可以在p中出现，否则会形成环
                if x not in p and (x+n==p[-1] or i+n==p[-1]):
                    affect_identifiability = True
                    break
            if not affect_identifiability:
                pop_list.insert(0, j)
            # if p[-1] not in pa_i+n and p[-1] != i+n:
            #     pop_list.insert(0,j)
        for j in range(len(pop_list)):
            paths[i].pop(pop_list[j])

def filter_remove(paths,x1,x2,path_id=-1):
    n=len(paths)
    changed=False
    # remove any path that contains x1->x2
    if path_id == -1:
        idx=range(n)
    else:
        idx=range(path_id,path_id+1)
    for i in idx:
        for j,p in enumerate(paths[i]):
            pop_list=[]
            if len(p) <2:
                continue
            for k in range(len(p)-1):
                if p[k]==x1 and p[k+1]==x2:
                    pop_list.insert(0,j)
                    changed=True
                    continue
            for k in range(len(pop_list)):
                paths[i].pop(pop_list[k])
    return paths,changed


def anm_orient(
    data: ndarray,
    alpha: float,
    FullmDAG,
    stable=True,
    show_progress: bool = False,
    datatype: str = 'linear'
):
    """
    Perform ANM discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    show_progress : True iff the algorithm progress should be show in console.
    stable : run stabilized skeleton discovery if True (default = True)
    verbose : True iff verbose output should be printed.

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    G = FullmDAG.copy()
    G=np.array(G)
    n=len(G)//2
    paths=[]

    for i in range(n):
        paths.append(find_R_descendent_paths(G, i))
    all_bi_edges=[]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if is_bi_edge(G,i,j):
                all_bi_edges.append((i,j))
    #G_bi_edges = all_bi_edges.copy()
    filter_related_path(G, paths, all_bi_edges)
    num_of_var = data.shape[1]


    pbar = tqdm(total=num_of_var) if show_progress else None

    Pa = {}
    pvalues=np.zeros(num_of_var)
    for x in range(num_of_var):
        if show_progress:
            pbar.reset()
        if show_progress:
            pbar.set_description(f'Depth=, working on node {x}')
        Pa[x] = [ ]
        Neigh_x = parent(G,x)
        if len(Neigh_x) <= 0:
            continue
        Pa_num = len(Neigh_x)
        found=False
        while Pa_num > 0:
            for S in combinations(Neigh_x, Pa_num):
                paths2=paths.copy()
                for p in S:
                    filter_remove(paths,x,p)
                pvalue=anm(x, S, data, datatype)
                if len(paths2[x])==0 and pvalue>=alpha: # satisfy anm model
                    Pa[x] = list(S)
                    pvalues[x]=pvalue
                    found=True
                    break
            if found:
                if not stable:
                    for p in Pa[x]:
                        G[p, x] = 1
                        G[x, p] = 0
                        filter_remove(paths,x,p)
                break
            Pa_num = Pa_num -1

        if show_progress:
            pbar.refresh()
    # print(Pa)
    if stable:
        for x in Pa.keys():
            for p in Pa[x]:
                if p in Pa and x in Pa[p]:
                    # if x->p
                    if pvalues[x]>pvalues[p]:
                        # p->x is better
                        G[p, x] = 1
                        G[x, p] = 0
                else:
                    G[p,x]=1
                    G[x,p]=0
    return G


def anm(x, S, data,datatype) -> bool:
    #1linear non-Gaussian or nolinear
    #linear non-Gaussian:
    #   linear regression
    #   hsic_test or kci_test
    #   nolinear:
    #   Gaussian_regression
    #   hsic_test or kci_test

    ## Learn regression models with test-wise deleted data
    involve_vars =  [x] + list(S)
    tdel_data = Helper.test_wise_deletion(data[:, involve_vars])
    X = tdel_data[:, 0]
    Pa_X = tdel_data[:, 1:]

    # linreg = LinearRegression().fit(Pa_X, X)
    # residuals = X - linreg.predict(Pa_X)
    if datatype == 'nonlinear':
        estimator = xgboost.XGBRegressor(objective="reg:squarederror",n_jobs=1, gamma=0)
        estimator.fit(Pa_X, X)
        X_predict=estimator.predict(Pa_X)
        residuals = X - X_predict
    # # gp = GaussianProcessRegressor().fit(Pa_X, X)
    # # X_predict = gp.predict(Pa_X)
    # residuals = X - X_predict
    elif datatype == 'linear':
        linreg = LinearRegression().fit(Pa_X, X)
        residuals = X - linreg.predict(Pa_X)
    kernelX = GaussianKernel()
    kernelY = GaussianKernel()
    # myspectralobject = HSICSpectralTestObject(Pa_X.shape[0], kernelX=kernelX, kernelY=kernelY,
    #                                               kernelX_use_median=True, kernelY_use_median=True,
    #                                               rff=True, num_rfx=20, num_rfy=20, num_nullsims=1000)
    # pvalue,hsic_stats,_ = myspectralobject.compute_pvalue_with_time_tracking(Pa_X.reshape((-1,Pa_X.shape[1])), residuals.reshape((-1,1)))
    # if pvalue < alpha:
    #     return False
    # return True

    myblockobject = HSICBlockTestObject(Pa_X.shape[0], kernelX=kernelX, kernelY=kernelY,
                                        kernelX_use_median=True, kernelY_use_median=True,
                                        blocksize=50, nullvarmethod='permutation')
    pvalue=myblockobject.compute_pvalue(Pa_X.reshape((-1,len(S))), residuals.reshape((-1,1)))
    # kci=KCI_UInd()
    # pvalue,test_stat=kci.compute_pvalue(Pa_X.reshape((-1,len(S))), residuals.reshape((-1,1)))
    return pvalue
