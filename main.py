from multiprocessing import Pool
#from multiprocessing_on_dill import Pool
import os
import pandas as pd
import numpy as np
import itertools
import networkx as nx
from collections.abc import Iterable
# Show progress bar
from tqdm import tqdm
# compute metrics
from castle.metrics import MetricsDAG
from copy import deepcopy
# Import tool function
from lib.skeleton_anm import skeleton_anm
from lib.utils import add_missing, DAG2ANME, DAG2mDAG, calculate_selfmasking_rate,randomMissGraph2
from lib.data import randomGraph,linear_data,nonlinear_data

from utils.utils import postprocess
def exp_missing(filename="sensitivity",seed=0, method='SM-MVPC', dim=15, indegree=1, sample_size=5000, sem_type="mlp",
                w_ranges=((-1.0, -0.5), (0.5, 1.0)), alpha=0.01, alpha2=0.001, missingness='SMAR', m_min=0.1, m_max=0.6, rom=1 / 3,
                num_self_node=2, quantile=0.2, skeleton=True, datatype='nonlinear', p=0.9,priority=0):
    """
    :param seed: The random seed.
    :param method: Select methods from 'SM-MVPC','LCS-MD','MVPC','TD-PC'
    :param dim: The numberb of nodes in graph
    :param indegree: The average indegree of the randomly generated graph
    :param sample_size: The number of sample size
    :param sem_type: The sem types for linear and nonlinear SEM. In Linear, the SEM types in linear include 'gauss,exp,sub-gauss,gumbel,uniform,logistic,poisson' for the choice of noise type. For the nonlinear SEM, SEM types includes 'mlp,mim,gp,gp-add' for different types of nonlinear function.
    :param w_ranges: The range of coefficient in linear SEM.
    :param alpha: The significance level (also called alpha) for the conditional independece test in learning causal skeleton.
    :param alpha2: The significance level (also called alpha) for the conditional independent test in learning causal direction using ANM
    :param missingness: The type of missingness mechanism types, including SMAR and SMNAR
    :param m_min: Control the missing probability before the quantile
    :param m_max: Control the missing probability after the quantile
    :param rom: Control the percentage of the number of missing variables.
    :param num_self_node: The number of weak self-masking variables.
    :param quantile: The quantile for dividing the missing probability into m_min and m_max, respectively.
    :param skeleton: Skeleton learning or causal structure learning.
    :param datatype: Select linear of nonlinear data.
    :param p: The parameter p controls the non-Gaussianity of the missingness: p = 1 gives a Gaussian, while p > 1 and p < 1 produces super-Gaussian and sub-Gaussian distributions respectively.
    :param indep_test: 'fisherz' 'mv_fisherz' 'mc_fisherz' 'kci' 'chisq' 'gsq' 'hsic_spectral'
    """

    randomState=np.random.RandomState(seed)

    B_true = randomGraph(dim, indegree, randomState)
    true_graph = nx.from_numpy_array(B_true, create_using=nx.DiGraph)
    # cause_dict: key是有缺失变量，例如1就是R_1，value是这个R_1的parent
    cause_dict=randomMissGraph2(true_graph, missingness=missingness, rom=rom, num_self_node=num_self_node, randomState=randomState)
    if cause_dict is None:
        return None
    if datatype == 'linear':
        data=linear_data(B_true, sample_size=sample_size, sem_type=sem_type, w_ranges=w_ranges, randomState=randomState, p=p)
    else:
        data = nonlinear_data(B_true, sample_size=sample_size, sem_type=sem_type, randomState=randomState)

    data_missing = add_missing(data, cause_dict=cause_dict, m_min=m_min, m_max=m_max,quantile=quantile,randomState=randomState)
    vars_miss = data_missing.columns[data_missing.isnull().any()].tolist()
    mDAG_learned = skeleton_anm(data_missing, alpha, alpha2, method=method, datatype = datatype, skeleton=skeleton,priority=priority)
    if method=="LCS-MD":
        mDAG_learned,mDAG_learned2=mDAG_learned
        # mDAG_learned2 is the results without apply rule
    if method=="MissDAG":
        # orient
        mDAG_true = DAG2mDAG(B_true.copy(), cause_dict)
        best_thres=0
        _, G_learned = postprocess(mDAG_learned.copy(), best_thres)
        mDAG = np.pad(G_learned, ((0, len(vars_miss)), (0, len(vars_miss))), 'constant')
        mt = MetricsDAG(mDAG, mDAG_true)
        min_SHD=mt.metrics['shd']
        ANME_true = DAG2ANME(B_true, cause_dict)

        bestG=mDAG.copy()
        for graph_thres in sorted(np.unique(mDAG_learned)):
            _, G_learned = postprocess(mDAG_learned.copy(), graph_thres)
            mDAG = np.pad(G_learned, ((0, len(vars_miss)), (0, len(vars_miss))), 'constant')
            mt = MetricsDAG(mDAG, mDAG_true)
            if mt.metrics['shd']<min_SHD:
                min_SHD=mt.metrics['shd']
                bestG = mDAG.copy()

        mt = MetricsDAG(bestG, ANME_true)
        mt2= MetricsDAG(bestG, mDAG_true)

        result = pd.DataFrame(
            columns=['seed', 'method', 'dim', 'Indegree', 'sample_size', 'sem_type', 'w_ranges', 'alpha','alpha2',
                     'missingness',
                     'm_min', 'm_max', 'rom', 'num_self_node', 'quantile', 'skeleton','priority', 'datatype', 'F1', 'SHD',
                     'Precision', 'Recall', 'p', 'F1_', 'SHD_', 'Precision_', 'Recall_', 'self_masking_rate',
                     'real_G','learned_G'])
        result.loc[len(result)] = [seed, method, dim, indegree, sample_size, sem_type, w_ranges, alpha, alpha2,
                                   missingness, m_min,
                                   m_max, rom, num_self_node, quantile, skeleton,priority, datatype, mt.metrics['F1'],
                                   mt.metrics['shd'],mt.metrics['precision'],mt.metrics['recall'], p,
                                   mt2.metrics['F1'],mt2.metrics['shd'], mt2.metrics['precision'], mt2.metrics['recall'], 0,
                                   mDAG_true.tolist(), mDAG_learned.tolist()]
        result.to_csv(f'result/orient/{filename}.csv', index=False, mode='a',
                      header=not os.path.exists(f'result/orient/{filename}.csv'))

        # skeleton
        B_ture_skeleton=np.array(B_true)
        B_ture_skeleton=B_ture_skeleton+B_ture_skeleton.T
        B_ture_skeleton=np.pad(B_ture_skeleton, ((0,len(cause_dict)),(0,len(cause_dict))), 'constant')

        R_true=np.zeros((dim+len(cause_dict),dim+len(cause_dict)))
        # Unify column order of R in mDAG_learned and mDAG_true
        R_list = data_missing.columns[data_missing.isnull().any()].tolist()
        for R in R_list:
            for Pa_R in cause_dict[R]:
                R_true[int(Pa_R)][R_list.index(R)+dim]=1
        mDAG_ture=B_ture_skeleton+R_true
        B_true_nd = np.triu(mDAG_ture, 1)

        best_thres=0
        _, G_learned = postprocess(mDAG_learned.copy(), best_thres)
        G_skeleton = G_learned + G_learned.T
        G_skeleton = np.pad(G_skeleton, ((0, len(vars_miss)), (0, len(vars_miss))), 'constant')
        # create R_learned
        R_learned = np.zeros((G_learned.shape[0] + len(vars_miss), G_learned.shape[1] + len(vars_miss)))
        mDAG = G_skeleton + R_learned
        mDAG = np.triu(mDAG, 1)
        mDAG[mDAG > 0] = 1
        mt = MetricsDAG(mDAG, B_true_nd)
        min_SHD=mt.metrics['shd']
        bestmt=deepcopy(mt)
        for graph_thres in sorted(np.unique(mDAG_learned)):
            _, G_learned = postprocess(mDAG_learned, graph_thres)
            G_skeleton = G_learned + G_learned.T
            G_skeleton = np.pad(G_skeleton, ((0, len(vars_miss)), (0, len(vars_miss))), 'constant')
            # create R_learned
            R_learned = np.zeros((G_learned.shape[0] + len(vars_miss), G_learned.shape[1] + len(vars_miss)))
            # mDAG
            mDAG = G_skeleton + R_learned
            mDAG = np.triu(mDAG, 1)
            mDAG[mDAG > 0] = 1
            mt = MetricsDAG(mDAG, B_true_nd)
            if mt.metrics['shd']<min_SHD:
                min_SHD=mt.metrics['shd']
                bestmt=deepcopy(mt)
                best_thres=graph_thres
        mt=bestmt
        result = pd.DataFrame(columns=['seed','method','dim','Indegree','sample_size','sem_type','w_ranges','alpha',
                                       'missingness', 'm_min', 'm_max','rom','num_self_node','quantile', 'skeleton','priority', 'datatype','F1', 'SHD', 'Precision', 'Recall','self_masking_rate',
                                       'real_G','learned_G'])
        result.loc[len(result)] = [seed, method, dim, indegree, sample_size, sem_type, w_ranges, alpha,
                                   missingness, m_min, m_max, rom, num_self_node, quantile, skeleton,priority, datatype, mt.metrics['F1'], mt.metrics['shd'], mt.metrics['precision'], mt.metrics['recall'],0,
                                   B_true_nd.tolist(), mDAG_learned.tolist()]
        result.to_csv(f'result/skeleton/{filename}.csv', index=False, mode='a', header=not os.path.exists(
            f'result/skeleton/{filename}.csv'))
        return result



    if skeleton:
        B_ture_skeleton=np.array(B_true)
        B_ture_skeleton=B_ture_skeleton+B_ture_skeleton.T
        B_ture_skeleton=np.pad(B_ture_skeleton, ((0,len(cause_dict)),(0,len(cause_dict))), 'constant')


        R_true=np.zeros((dim+len(cause_dict),dim+len(cause_dict)))
        # Unify column order of R in mDAG_learned and mDAG_true
        R_list = data_missing.columns[data_missing.isnull().any()].tolist()
        for R in R_list:
            for Pa_R in cause_dict[R]:
                R_true[int(Pa_R)][R_list.index(R)+dim]=1
        mDAG_ture=B_ture_skeleton+R_true

        B_true_nd = np.triu(mDAG_ture, 1)
        mDAG_learned = np.triu(mDAG_learned, 1)


        mt = MetricsDAG(mDAG_learned, B_true_nd)
        self_masking_rate = calculate_selfmasking_rate(mDAG_learned, cause_dict)

        result = pd.DataFrame(columns=['seed','method','dim','Indegree','sample_size','sem_type','w_ranges','alpha',
                                       'missingness', 'm_min', 'm_max','rom','num_self_node','quantile', 'skeleton','priority', 'datatype','F1', 'SHD', 'Precision', 'Recall','self_masking_rate',
                                       'real_G','learned_G'])
        result.loc[len(result)] = [seed, method, dim, indegree, sample_size, sem_type, w_ranges, alpha,
                                   missingness, m_min, m_max, rom, num_self_node, quantile, skeleton,priority, datatype, mt.metrics['F1'], mt.metrics['shd'], mt.metrics['precision'], mt.metrics['recall'],self_masking_rate,
                                   B_true_nd.tolist(), mDAG_learned.tolist()]
        result.to_csv(f'result/skeleton/{filename}.csv', index=False, mode='a', header=not os.path.exists(
            f'result/skeleton/{filename}.csv'))
        return result
    else:

        ANME_true = DAG2ANME(B_true, cause_dict)

        mDAG_true = DAG2mDAG(B_true, cause_dict)
        mt = MetricsDAG(mDAG_learned, ANME_true)
        mt2 = MetricsDAG(mDAG_learned, mDAG_true)
        # Calculate selfmask_ rate
        self_masking_rate = calculate_selfmasking_rate(mDAG_learned, cause_dict)


        # save result
        result = pd.DataFrame(
            columns=['seed', 'method', 'dim', 'Indegree', 'sample_size', 'sem_type', 'w_ranges', 'alpha','alpha2',
                     'missingness',
                     'm_min', 'm_max', 'rom', 'num_self_node', 'quantile', 'skeleton','priority', 'datatype', 'F1', 'SHD',
                     'Precision', 'Recall', 'p', 'F1_', 'SHD_', 'Precision_', 'Recall_', 'self_masking_rate',
                     'real_G','learned_G'])
        result.loc[len(result)] = [seed, method, dim, indegree, sample_size, sem_type, w_ranges, alpha, alpha2,
                                   missingness, m_min,
                                   m_max, rom, num_self_node, quantile, skeleton,priority, datatype, mt.metrics['F1'],
                                   mt.metrics['shd'],mt.metrics['precision'],mt.metrics['recall'], p,
                                   mt2.metrics['F1'],mt2.metrics['shd'], mt2.metrics['precision'], mt2.metrics['recall'], self_masking_rate,
                                   mDAG_true.tolist(), mDAG_learned.tolist()]
        result.to_csv(f'result/orient/{filename}.csv', index=False, mode='a',
                      header=not os.path.exists(f'result/orient/{filename}.csv'))

        if method == "LCS-MD":
            mt3 = MetricsDAG(mDAG_learned2, ANME_true)
            mt4 = MetricsDAG(mDAG_learned2, mDAG_true)
            result = pd.DataFrame(
                columns=['seed', 'method', 'dim', 'Indegree', 'sample_size', 'sem_type', 'w_ranges', 'alpha', 'alpha2',
                         'missingness',
                         'm_min', 'm_max', 'rom', 'num_self_node', 'quantile', 'skeleton', 'priority', 'datatype', 'F1',
                         'SHD',
                         'Precision', 'Recall', 'p', 'F1_', 'SHD_', 'Precision_', 'Recall_', 'self_masking_rate',
                         'real_G', 'learned_G'])
            result.loc[len(result)] = [seed, "SM-MVPC-ANM", dim, indegree, sample_size, sem_type, w_ranges, alpha, alpha2,
                                       missingness, m_min,
                                       m_max, rom, num_self_node, quantile, skeleton, priority, datatype,
                                       mt3.metrics['F1'],
                                       mt3.metrics['shd'], mt3.metrics['precision'], mt3.metrics['recall'], p,
                                       mt4.metrics['F1'], mt4.metrics['shd'], mt4.metrics['precision'],
                                       mt4.metrics['recall'], self_masking_rate,
                                       mDAG_true.tolist(), mDAG_learned.tolist()]
            result.to_csv(f'result/orient/{filename}.csv', index=False, mode='a',
                          header=not os.path.exists(f'result/orient/{filename}.csv'))

        return result

if __name__=="__main__":

    # parameter settings
    filename="sensitivity"

    datatype='nonlinear'
    sem_type='mim'
    missingness='SMNAR'
    seed=1
    sample_size=7000
    in_degree=1
    num_self_node=3
    dim=10

    p = 0.9
    alpha = 0.01
    alpha2 = 0.01
    priority=-1

    exp_missing(filename=filename,seed=seed, method='LCS-MD', dim=dim, indegree=in_degree, sample_size=sample_size, sem_type=sem_type, w_ranges=((-3.0, -1.0), (1.0, 3.0)), alpha=alpha, alpha2=alpha2,missingness=missingness, m_min=0.1, m_max=0.7,rom= 1/3, num_self_node=2, quantile=0.2, skeleton=False,datatype=datatype, p=p,priority=priority)
    
    exp_missing(filename=filename,seed=seed, method='SM-MVPC', dim=dim, indegree=in_degree, sample_size=sample_size, sem_type=sem_type, w_ranges=((-3.0, -1.0), (1.0, 3.0)), alpha=alpha, alpha2=alpha2,missingness=missingness, m_min=0.1, m_max=0.7,rom= 1/3, num_self_node=2, quantile=0.2, skeleton=True,datatype=datatype, p=p,priority=priority)
