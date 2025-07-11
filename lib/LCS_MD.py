from __future__ import annotations
import networkx as nx
from numpy import ndarray
from copy import deepcopy

from itertools import combinations, permutations
from typing import Dict, List
from causallearn2.graph.GraphClass import CausalGraph
from causallearn2.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn2.utils.cit import *
from causallearn2.utils.PCUtils import Helper, SkeletonDiscovery, UCSepset
from causallearn2.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge
from causallearn2.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
####import my
from causallearn2.utils.cit import CIT
from lib.utils import cg2FullmDAG

from lib.orient import anm_orient

from lib.ANM_rule_path import anm_rule_by_paths

def get_missingness_index(data: ndarray) -> List[int]:
    """Detect the parents of missingness indicators
    :param data: data set (numpy ndarray)
    :return:
    missingness_index: list, the index of missingness indicators
    """

    missingness_index = []
    _, ncol = np.shape(data)
    for i in range(ncol):
        if np.isnan(data[:, i]).any():
            missingness_index.append(i)
    return missingness_index

def isempty(prt_r) -> bool:
    """Test whether the parent of a missingness indicator is empty"""
    return len(prt_r) == 0

def get_parent_missingness_pairs(data: ndarray, alpha: float, indep_test, stable: bool = True) -> Dict[str, list]:
    """
    Detect the parents of missingness indicators
    If a missingness indicator has no parent, it will not be included in the result
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param indep_test: name of the test-wise deletion independence test being used
        - "MV_Fisher_Z": Fisher's Z conditional independence test
        - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """
    parent_missingness_pairs = {'prt': [], 'm': []}

    ## Get the index of missingness indicators
    missingness_index = get_missingness_index(data)

    ## Get the index of parents of missingness indicators
    # If the missingness indicator has no parent, then it will not be collected in prt_m
    for missingness_i in missingness_index:
        parent_of_missingness_i = detect_parent(missingness_i, data, alpha, indep_test, stable)
        if not isempty(parent_of_missingness_i):
            parent_missingness_pairs['prt'].append(parent_of_missingness_i)
            parent_missingness_pairs['m'].append(missingness_i)
    return parent_missingness_pairs


def detect_parent(r: int, data_: ndarray, alpha: float, indep_test, stable: bool = True) -> ndarray:
    """Detect the parents of a missingness indicator
    :param r: the missingness indicator
    :param data_: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param indep_test: name of the test-wise deletion independence test being used
        - "MV_Fisher_Z": Fisher's Z conditional independence test
        - "MV_G_sq": G-squared conditional independence test (TODO: under development)
    :param stable: run stabilized skeleton discovery if True (default = True)
    : return:
    prt: parent of the missingness indicator, r
    """
    ## TODO: in the test-wise deletion CI test, if test between a binary and a continuous variable,
    #  there can be the case where the binary variable only take one value after deletion.
    #  It is because the assumption is violated.

    ## *********** Adaptation 0 ***********
    # For avoid changing the original data
    data = data_.copy()
    ## *********** End ***********

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    ## *********** Adaptation 1 ***********
    # data
    ## Replace the variable r with its missingness indicator
    ## If r is not a missingness indicator, return [].
    data[:, r] = np.isnan(data[:, r]).astype(float)  # True is missing; false is not missing
    if sum(data[:, r]) == 0 or sum(data[:, r]) == len(data[:, r]):
        return np.empty(0)
    ## *********** End ***********

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var)
    cg.set_ind_test(CIT(data, indep_test.method))

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:

            ## *********** Adaptation 2 ***********
            # the skeleton search
            ## Only test which variable is the neighbor of r
            if x != r:
                continue
            ## *********** End ***********

            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            Helper.append_value(cg.sepset, x, y, S)
                            Helper.append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    ## *********** Adaptation 3 ***********
    ## extract the parent of r from the graph
    cg.to_nx_skeleton()
    cg_skel_adj = nx.to_numpy_array(cg.nx_skel).astype(int)
    prt = get_parent(r, cg_skel_adj)
    ## *********** End ***********

    return prt

def get_parent(r: int, cg_skel_adj: ndarray) -> ndarray:
    """Get the neighbors of missingness indicators which are the parents
    :param r: the missingness indicator index
    :param cg_skel_adj: adjacancy matrix of a causal skeleton
    :return:
    prt: list, parents of the missingness indicator r
    """
    num_var = len(cg_skel_adj[0, :])
    indx = np.array([i for i in range(num_var)])
    prt = indx[cg_skel_adj[r, :] == 1]
    return prt

def is_self_masking_node(cg_pre, prt_m, var_miss) -> bool:
    vars_miss = prt_m['m']
    col = vars_miss.index(var_miss)
    Pa_R = [prt_m['prt'][col][i] for i in range(len(prt_m['prt'][col]))]
    num_vars = cg_pre.G.num_vars
    for (i, j) in combinations(range(num_vars), 2):
        learned_sepset = cg_pre.sepset[i, j][-1]
        if len(learned_sepset) > 0 and var_miss in learned_sepset and i in Pa_R and j in Pa_R:
            return True
    return False

def get_self_masking_parent_missingness(cg_pre, prt_m) -> Dict[str, list]:
    vars_miss = prt_m['m']
    for var_miss in vars_miss:
        if is_self_masking_node(cg_pre, prt_m, var_miss):
            # Reset the parent that resulted in the missing node
            col = vars_miss.index(var_miss)
            prt_m['prt'][col] = np.array([var_miss])

    return prt_m

def skeleton_correction(data: ndarray, alpha: float, test_with_correction_name: str, init_cg: CausalGraph, prt_m: dict,
                        stable: bool = True) -> CausalGraph:
    """Perform skeleton discovery
    :param data: data set (numpy ndarray)
    :param alpha: desired significance level in (0, 1) (float)
    :param test_with_correction_name: name of the independence test being used
           - "MV_Crtn_Fisher_Z": Fisher's Z conditional independence test
           - "MV_Crtn_G_sq": G-squared conditional independence test
    :param stable: run stabilized skeleton discovery if True (default = True)
    :return:
    cg: a CausalGraph object
    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1
    assert test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]

    ## *********** Adaption 1 ***********
    no_of_var = data.shape[1]

    ## Initialize the graph with the result of test-wise deletion skeletion search
    cg = init_cg

    if test_with_correction_name in ["MV_Crtn_Fisher_Z", "MV_Crtn_G_sq"]:
        cg.set_ind_test(CIT(data, "mc_fisherz"))
    # No need of the correlation matrix if using test-wise deletion test
    cg.prt_m = prt_m
    ## *********** Adaption 1 ***********

    node_ids = range(no_of_var)
    pair_of_variables = list(permutations(node_ids, 2))

    depth = -1
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        for (x, y) in pair_of_variables:
            Neigh_x = cg.neighbors(x)
            if y not in Neigh_x:
                continue
            else:
                Neigh_x = np.delete(Neigh_x, np.where(Neigh_x == y))

            if len(Neigh_x) >= depth:
                for S in combinations(Neigh_x, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if not stable:  # Unstable: Remove x---y right away
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                        else:  # Stable: x---y will be removed only
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered
                            Helper.append_value(cg.sepset, x, y, S)
                            Helper.append_value(cg.sepset, y, x, S)
                        break

        for (x, y) in list(set(edge_removal)):
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)

    return cg

def SM_MVPC(data,
            alpha,
            indep_test: str,
            stable: bool,
            node_names=None,
            verbose: bool = False,
            show_progress: bool = False,
            correction_name: str = 'MV_Crtn_Fisher_Z',
            background_knowledge: BackgroundKnowledge | None = None):


    indep_test_cit = CIT(data, indep_test)

    ## Step 1: detect the direct causes of missingness indicators
    prt_m = get_parent_missingness_pairs(data, alpha, indep_test_cit, stable)
    # print('Finish detecting the parents of missingness indicators.  ')

    ## Step 2:
    ## a) Run PC algorithm with the 1st step skeleton;
    cg = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test_cit, stable,
                                                  background_knowledge=background_knowledge,
                                                  verbose=verbose, show_progress=show_progress, node_names=node_names)
    if background_knowledge is not None:
        orient_by_background_knowledge(cg, background_knowledge)

    cg.to_nx_skeleton()
    # print('Finish skeleton search with test-wise deletion.')

    ###### rewrite my code find self-masking node
    prt_m = get_self_masking_parent_missingness(cg, prt_m)
    # print(prt_m)
    cg.prt_m = prt_m
    ## b) Correction of the extra edges
    if indep_test!="kci":
        cg = skeleton_correction(data, alpha, correction_name, cg, prt_m, stable)
        prt_m = get_self_masking_parent_missingness(cg, prt_m)
    # print(prt_m)
    # print('Finish missingness correction.')

    if background_knowledge is not None:
        orient_by_background_knowledge(cg, background_knowledge)


    return cg

def pc_orient(cg_1,alpha,uc_rule=0,uc_priority=2):
    if uc_rule == 0:
        if uc_priority != -1:
            cg_2 = UCSepset.uc_sepset(cg_1, uc_priority)
        else:
            cg_2 = UCSepset.uc_sepset(cg_1)
        cg = Meek.meek(cg_2)

    elif uc_rule == 1:
        if uc_priority != -1:
            cg_2 = UCSepset.maxp(cg_1, uc_priority)
        else:
            cg_2 = UCSepset.maxp(cg_1)
        cg = Meek.meek(cg_2)

    elif uc_rule == 2:
        if uc_priority != -1:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha, uc_priority)
        else:
            cg_2 = UCSepset.definite_maxp(cg_1, alpha)
        cg_before = Meek.definite_meek(cg_2)
        cg = Meek.meek(cg_before)
    else:
        raise ValueError("uc_rule should be in [0, 1, 2]")
    return cg

def is_bi_edge(G, i, j):
    if G[i, j] == 1 and G[j, i] == 1:
        return True
    else:
        return False
def is_connect(G,i,j):
    if G[i,j]!=0 or G[j,i]!=0:
        return True
    return False
def combine_cpdag(G,cpdag,priority=0):
    # priority=0: G has priority
    # priority=1: cpdag has priority
    assert len(G)==len(cpdag)
    n=len(G)//2
    G=G.copy()
    for i in range(n):
        for j in range(n):
            if is_bi_edge(G,i,j) and not is_bi_edge(cpdag,i,j):
                G[i,j]=cpdag[i,j]
                G[j,i]=cpdag[j,i]
            elif priority==1 and is_connect(cpdag,i,j) and not is_bi_edge(cpdag,i,j):
                # priority=1: cpdag has priority
                G[i,j]=cpdag[i,j]
                G[j,i]=cpdag[j,i]
    return G

def lcs_md(data,indep_test, varnames, vars_miss,  alpha = 0.01, alpha2 = 0.01, datatype = 'linear',apply_rule=True, priority=0):
    """
    apply_rule: whether apply the anm rule
    priority=[-1,0,1] -1: does not consider v-structure. 0: result from anm has the priority 1: v-structure has the priority if there exist conflict.
    """
    mvpc_cg = SM_MVPC(data, alpha, indep_test, True, varnames,False)
    if priority!=-1:
        oriented_cg = pc_orient(deepcopy(mvpc_cg), alpha)
        cpdag = cg2FullmDAG(oriented_cg, vars_miss)


    FullmDAG = anm_orient(data, alpha2, cg2FullmDAG(mvpc_cg,vars_miss), datatype = datatype)
    #  check cyc

    # adj convent to FullmDAG
    #FullmDAG = cg2FullmDAG(cg, vars_miss)
    # FullmDAG convent to   format  x--y   1--1  x-->y   0--1
    SM_MVPC_ANM = FullmDAG.copy() # This one will not apply rule
    # rule
    if apply_rule:
        FullmDAG, paths = anm_rule_by_paths(FullmDAG.copy())

    if priority!=-1:
        FullmDAG=combine_cpdag(FullmDAG,cpdag,priority)
        SM_MVPC_ANM=combine_cpdag(SM_MVPC_ANM,cpdag,priority)

    indexSelect = list(range(len(varnames)))
    for R_i in vars_miss:
        R_index = len(varnames) + int(R_i)
        indexSelect.append(R_index)
    mDAG = FullmDAG[np.ix_(indexSelect, indexSelect)]
    SM_MVPC_ANM = SM_MVPC_ANM[np.ix_(indexSelect, indexSelect)]

    return mDAG, SM_MVPC_ANM


