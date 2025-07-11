import numpy as np
import numba
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product

from numba.typed import List, Dict
from numba import types, typed, typeof
import pandas as pd
from networkx.drawing.nx_pydot import graphviz_layout
from itertools import combinations
#from copy import deepcopy
@numba.njit(cache=True)
def is_bi_edge(G, i, j):
    if G[i, j] == 1 and G[j, i] == 1:
        return True
    else:
        return False

@numba.njit
def deepcopy(paths):
    new_paths=[]
    for node_paths in paths:
        new_paths.append(node_paths.copy())
    return new_paths

@numba.njit
def AssignList(paths,new_paths):
    # paths = new_paths
    if len(paths)!=len(new_paths):
        raise NotImplementedError
    for i in range(len(paths)):
        paths[i]=new_paths[i]



@numba.njit(cache=True)
def child(G, i, bi_direct=True):
    if bi_direct:
        return np.flatnonzero(G[i, :] == 1)
    else:
        res = []
        ch = np.flatnonzero(G[i, :] == 1)
        for v in ch:
            if not is_bi_edge(G, i, v):
                res.append(v)
        return np.array(res)


@numba.njit(cache=True)
def parent(G, i, bi_direct=True):
    if bi_direct:
        return np.flatnonzero(G[:, i] == 1)
    else:
        res = []
        pa = np.flatnonzero(G[:, i] == 1)
        for v in pa:
            if not is_bi_edge(G, v, i):
                res.append(v)
        return np.array(res)

@numba.njit
def descendent(G,i,des,bi_direct=False,include_missing_indicator=False):
    ch_i=child(G,i,bi_direct)
    n=len(G)//2
    for c in ch_i:
        if not include_missing_indicator:
            if c < n and c not in des:
                des.add(c)
                descendent(G,c,des,bi_direct,include_missing_indicator)
        else:
            if c not in des:
                des.add(c)
                descendent(G,c,des,bi_direct,include_missing_indicator)
    return des

@numba.njit
def is_cycle(G, i, j, path=[]):  # 判断是否必然形成环
    if G[i, j] != 1:
        return False
    path.append(i)
    # pdb.set_trace()
    if j in path:
        return True
    for c in child(G, j):
        if is_bi_edge(G, j, c):
            continue
        if is_cycle(G, j, c, path):
            return True
        path.pop()
    return False

@numba.njit(cache=True)
def ancestor(G, i, anc, bi_direct=False):
    pa=parent(G, i, bi_direct=bi_direct)
    for p in pa:
        if p not in anc:
            anc.add(p)
            ancestor(G,p,anc,bi_direct=bi_direct)
    return anc

@numba.njit
def find_R_descendent_paths(G, i, stk=None, ans=None):
    # https://leetcode-cn.com/problems/all-paths-from-source-to-target/solution/suo-you-ke-neng-de-lu-jing-by-leetcode-s-iyoh/
    if stk is None and ans is None:
        stk = []
        stk.append(i)
        ans = []
        ans.append(stk)
        ans.pop()
    n = len(G) // 2

    if i >= n:
        ans.append(stk.copy())
        return

    for y in child(G,i):
        if y not in stk:
            stk.append(y)
            find_R_descendent_paths(G, y, stk, ans)
            stk.pop()

    return ans

@numba.njit
def is_affect_identifiability(n,path,pa,i):
    # for a given path in node i, tell whether pa can be the parent of this path.
    if pa not in path and (pa + n == path[-1] or i + n == path[-1]):
        # if pa is not in the path and the indicator is R_pa or R_i then pa is the possible node of this path.
        return True
    else:
        return False

@numba.njit
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

@numba.njit
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

@numba.njit
def RULE1(G,paths,all_bi_edges,i,j):
    # filter the path that cause cycle
    changed = False
    G[i, j] = 1  # i->j
    G[j, i] = 0  # i->j
    i2j = is_cycle(G, i, j, List.empty_list(types.int64))
    G[i, j] = 0  # j->i
    G[j, i] = 1  # j->i
    j2i = is_cycle(G, j, i, List.empty_list(types.int64))
    if i2j:  # 如果i->j必然形成环，则意味着j->i成立
        G[j, i] = 1
        G[i, j] = 0
        changed=True
        filter_if_orient(G,paths,all_bi_edges,j,i)
        #filter_remove(paths,i,j)
        return changed
    if j2i:
        G[j, i] = 0
        G[i, j] = 1
        changed = True
        filter_if_orient(G, paths, all_bi_edges, i, j)
        #filter_remove(paths, j, i)
        return changed
    G[j, i] = 1
    G[i, j] = 1
    return changed

@numba.njit
def RULE2(G,paths,all_bi_edges,i,j):
    # filter the path that cause cycle
    #n = len(G) // 2

    #changed=False
    G1=G.copy()
    path_i2j=deepcopy(paths)
    G1[i, j] = 1  # i->j
    G1[j, i] = 0  # i->j
    filter_if_orient(G1,path_i2j,all_bi_edges,i,j)
    orient_by_RULE1(G1, path_i2j, all_bi_edges)
    #path_i2j=filter_remove(deepcopy(paths),j,i,path_id=j)

    G2=G.copy()
    path_j2i=deepcopy(paths)
    G2[i, j] = 0  # j->i
    G2[j, i] = 1  # j->i
    #path_j2i=filter_remove(deepcopy(paths),i,j,path_id=i)
    filter_if_orient(G2, path_j2i, all_bi_edges, j, i)
    orient_by_RULE1(G2, path_j2i, all_bi_edges)

    for a, b in all_bi_edges:
        if (G1[a,b]==1 and G1[b,a]==0 and is_anm_identifiable_by_path(G1,path_i2j,b)) or (G1[a,b]==0 and G1[b,a]==1 and is_anm_identifiable_by_path(G1,path_i2j,a)):
            # can not point to an identifiable path
            # G[i, j] = 0
            # G[j, i] = 1
            AssignList(G, G2)
            AssignList(paths, path_j2i)
            # paths=path_j2i
            return True
        elif (G2[a, b] == 1 and G2[b, a] == 0 and is_anm_identifiable_by_path(G2, path_j2i, b)) or (G2[a, b] == 0 and G2[b, a] == 1 and is_anm_identifiable_by_path(G2, path_j2i, a)):
            # b->a
            # can not point to an identifiable path
            # G[j, i] = 0
            # G[i, j] = 1
            AssignList(G, G1)
            AssignList(paths, path_i2j)
            # filter_remove(paths, j, i)
            return True
    G[j, i] = 1
    G[i, j] = 1
    return False

@numba.njit
def orient_by_RULE1(G,paths,all_bi_edges):
    n = len(G) // 2
    changed = True
    G_bi_edges=[]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if is_bi_edge(G,i,j):
                G_bi_edges.append((i,j))
    while changed:
        changed = False
        for i,j in G_bi_edges:
            cycle_rule1=RULE1(G,paths,all_bi_edges, i, j)
            if cycle_rule1:
                changed=True
                G_bi_edges.remove((i,j))
                continue
    return G

@numba.njit
def orient_by_RULE1_and_RULE2(G,paths,all_bi_edges):
    n = len(G) // 2
    changed = True
    G_bi_edges=[]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if is_bi_edge(G,i,j):
                G_bi_edges.append((i,j))
    while changed:
        changed = False
        for i,j in G_bi_edges:
            cycle_rule1=RULE1(G,paths,all_bi_edges, i, j)
            if cycle_rule1:
                changed=True
                G_bi_edges.remove((i,j))
                continue
            anm_rule2 = RULE2(G,paths,all_bi_edges, i, j)
            if anm_rule2:
                changed=True
                G_bi_edges.remove((i, j))
                continue
    return G

@numba.njit
def RULE3(G,paths,all_bi_edges,i,j):
    # Detecting whether conflict exist if we orient i->j.
    G1=G.copy()
    paths_i2j=deepcopy(paths)
    G1[i,j]=1
    G1[j,i]=0
    paths_i2j=filter_if_orient(G1,paths_i2j,all_bi_edges,i,j)
    orient_by_RULE1_and_RULE2(G1,paths_i2j,all_bi_edges)
    for a, b in all_bi_edges:
        if (G1[a,b]==1 and G1[b,a]==0 and is_anm_identifiable_by_path(G1,paths_i2j,b)) or (G1[a,b]==0 and G1[b,a]==1 and is_anm_identifiable_by_path(G1,paths_i2j,a)):
            # can not point to an identifiable path
            G[i, j] = 0
            G[j, i] = 1
            filter_if_orient(G, paths, all_bi_edges, j, i)
            return True
    changed=False
    for a, b in all_bi_edges:
        if (i, j) != (a, b) and (j, i) != (a, b):
            a2b = G1[a, b]
            b2a = G1[b, a]
            G1[a, b] = 1
            G1[b, a] = 0
            if is_cycle(G1,a,b,List.empty_list(types.int64)):
                G1[a, b] = a2b
                G1[b, a] = b2a
                continue
            path_a2b = filter_if_orient(G1, deepcopy(paths_i2j), all_bi_edges, a, b)
            anm_a2b = is_anm_identifiable_by_path(G1, path_a2b, b)

            G1[a, b] = 0
            G1[b, a] = 1
            if is_cycle(G1,b,a,List.empty_list(types.int64)):
                G1[a, b] = a2b
                G1[b, a] = b2a
                continue
            path_b2a = filter_if_orient(G1, deepcopy(paths_i2j), all_bi_edges, b, a)
            anm_b2a = is_anm_identifiable_by_path(G1, path_b2a, a)
            G1[a, b] = a2b
            G1[b, a] = b2a
            if anm_a2b and anm_b2a:
                G[i, j] = 0
                G[j, i] = 1
                filter_if_orient(G, paths, all_bi_edges, j, i)
                # print(f"orient {j}->{i} because {a}-{b} appears to be both anm identifable if {i}->{j}")
                changed = True
                break
    return changed

@numba.njit
def filter_cycle_path(G,paths,i,j):
    # if we orient i->j than we need to remove all paths that exists j->...->i
    changed=False
    des=set()
    des.add(j)
    descendent(G,j,des,bi_direct=False,include_missing_indicator=False)
    anc=set()
    anc.add(i)
    ancestor(G,i,anc,bi_direct=False)
    for j in des:
        if i not in child(G,j):
            # 如果i和j是相邻的那这个不能算是环
            for node_path in paths:
                pop_list=[]
                for idx,p in enumerate(node_path):
                    finded_j=False
                    for k in range(len(p)):
                        if not finded_j and p[k]==j:
                            finded_j=True
                        elif finded_j and p[k] in anc:
                            pop_list.insert(0,idx)
                            changed = True
                            break
                for k in range(len(pop_list)):
                    node_path.pop(pop_list[k])

    return paths, changed

@numba.njit
def is_anm_identifiable_by_path(G, paths, i):
    if len(paths[i])==0:
        return True
    # pa_i=parent(G,i)
    # for x in pa_i:
    #     if is_bi_edge(G,x,i):
    #         r_path,_=filter_remove(deepcopy(paths),i,x,path_id=i) # x->i
    #         if len(r_path[i])==0:
    #             return True
    return False


@numba.njit
def filter_conflict_path(G, paths, all_bi_edges, i, j):
    #n=len(G)//2
    #G_backup=G.copy()
    #paths_backup=deepcopy(paths)
    changed=False

    for a, b in all_bi_edges:
        if is_bi_edge(G, a, b):
            if not is_anm_identifiable_by_path(G,paths,b):
                # orient a->b
                G[a, b]=1
                G[b, a]=0
                r_paths,_=filter_remove(deepcopy(paths), b, a)
                filter_related_path(G, r_paths,all_bi_edges, a)
                filter_related_path(G, r_paths,all_bi_edges, b)
                filter_cycle_path(G,r_paths, a, b)
                for k,l in all_bi_edges:
                    if (a, b)!=(k, l) and (i,j)!=(k,l) and (j,i)!=(k,l):
                        if not is_bi_edge(G,k,l):
                            if G[k,l]==1:
                                anm_l=is_anm_identifiable_by_path(G, r_paths, l)
                                r_paths_k,_ = filter_remove(deepcopy(r_paths), k, l)
                                anm_k = is_anm_identifiable_by_path(G, r_paths_k, k)
                            else:
                                anm_k = is_anm_identifiable_by_path(G, r_paths, k)
                                r_paths_l,_ = filter_remove(deepcopy(r_paths), l, k)
                                anm_l = is_anm_identifiable_by_path(G, r_paths_l, l)
                        else:
                            r_paths_k,_ = filter_remove(deepcopy(r_paths), k, l)
                            anm_k=is_anm_identifiable_by_path(G, r_paths_k, k)
                            r_paths_l,_ = filter_remove(deepcopy(r_paths), l, k)
                            anm_l = is_anm_identifiable_by_path(G, r_paths_l, l)
                        if anm_k and anm_l:
                            # remove a->b
                            #print(f"{k} and {l} confict when {a}->{b}")
                            G[a, b] = 0
                            G[b, a] = 1
                            paths,state=filter_remove(paths, a, b)
                            if state:
                                changed=True
                            #changed=True
                            #continue
            G[a, b] = 1
            G[b, a] = 1
            if not is_anm_identifiable_by_path(G,paths,a):
                # orient b->a
                G[a, b]=0
                G[b, a]=1
                r_paths,_ = filter_remove(deepcopy(paths), a, b)
                filter_related_path(G, r_paths,all_bi_edges, a)
                filter_related_path(G, r_paths,all_bi_edges, b)
                filter_cycle_path(G,r_paths, b, a)
                for k, l in all_bi_edges:
                    if (a, b) != (k, l):
                        if not is_bi_edge(G, k, l):
                            if G[k, l] == 1:
                                anm_l = is_anm_identifiable_by_path(G, r_paths, l)
                                r_paths_k,_ = filter_remove(deepcopy(r_paths), k, l)
                                anm_k = is_anm_identifiable_by_path(G, r_paths_k, k)
                            else:
                                anm_k = is_anm_identifiable_by_path(G, r_paths, k)
                                r_paths_l,_ = filter_remove(deepcopy(r_paths), l, k)
                                anm_l = is_anm_identifiable_by_path(G, r_paths_l, l)
                        else:
                            r_paths_k,_ = filter_remove(deepcopy(r_paths), k, l)
                            anm_k = is_anm_identifiable_by_path(G, r_paths_k, k)
                            r_paths_l,_ = filter_remove(deepcopy(r_paths), l, k)
                            anm_l = is_anm_identifiable_by_path(G, r_paths_l, l)
                        if anm_k and anm_l:
                            # remove b->a
                            #print(f"{k} and {l} confict when {b}->{a}")
                            G[a, b] = 1
                            G[b, a] = 0
                            paths ,state= filter_remove(paths, b, a)
                            if state:
                                changed = True
                            #changed = True
                            #continue
            G[a, b] = 1
            G[b, a] = 1
    return paths,changed

@numba.njit
def filter_identifiable_path(G,paths,all_bi_edges):
    n=len(paths)
    changed=False
    for i in range(n):
        if is_anm_identifiable_by_path(G,paths,i):
            pa_i=parent(G,i)
            for x in pa_i:
                if is_bi_edge(G,x,i):
                    # x can not point to i
                    G[x,i]=0
                    paths,state=filter_remove(paths,x,i)
                    filter_related_path(G, paths,all_bi_edges, i)
                    filter_related_path(G, paths,all_bi_edges, x)
                    filter_cycle_path(G, paths, i, x)
                    if state:
                        changed=True
                    #changed=True
    return paths,changed


@numba.njit
def filter_if_orient(G,paths,all_bi_edges,i,j):
    # if we orient i->j than we need to filter all paths to avoid the circle path
    # and remove all path that contains conflict
    G=G.copy()
    filter_remove(paths,j,i)
    filter_related_path(G,paths,all_bi_edges,i)
    filter_related_path(G,paths,all_bi_edges,j)
    filter_cycle_path(G,paths, i, j)
    changed = True
    while changed:
        changed=False
        paths, changed1 = filter_conflict_path(G,paths,all_bi_edges,i,j)
        paths, changed2 = filter_identifiable_path(G,paths,all_bi_edges)
        if changed1 or changed2:
            changed=True
    return paths

@numba.njit
def anm_rule_by_paths(G):
    n=len(G)//2
    paths=[]
    for i in range(n):
        paths.append(find_R_descendent_paths(G, i))
    all_bi_edges=[]
    for i in range(n - 1):
        for j in range(i + 1, n):
            if is_bi_edge(G,i,j):
                all_bi_edges.append((i,j))
    G_bi_edges = all_bi_edges.copy()
    filter_related_path(G, paths, all_bi_edges)
    changed=True
    while changed:
        changed = False
        for i,j in G_bi_edges:
            cycle_rule1=RULE1(G,paths,all_bi_edges, i, j)
            if cycle_rule1:
                changed=True
                G_bi_edges.remove((i,j))
                continue
            anm_rule2 = RULE2(G,paths,all_bi_edges, i, j)
            if anm_rule2:
                changed=True
                G_bi_edges.remove((i, j))
                continue
            conflict_rule_i2j=RULE3(G,paths,all_bi_edges,i,j)
            if conflict_rule_i2j:
                changed=True
                G_bi_edges.remove((i, j))
                continue
            conflict_rule_j2i=RULE3(G,paths,all_bi_edges,j,i)
            if conflict_rule_j2i:
                changed=True
                G_bi_edges.remove((i, j))
                continue
            # 所有规则都无法判断则恢复原状
            G[i, j] = 1
            G[j, i] = 1
    #G2,paths2=final_check(G.copy(), deepcopy(paths),all_bi_edges)
    return G,paths



def gen_G(XX, XR):
    XX = np.array(XX)
    XR = np.array(XR)
    #max_x = np.max(XX)
    n = np.max(XX) + 1
    #min_x = np.min(XX)
    #n=max_x-min_x+1
    # n_missing=len(np.unique(XR[:,1]))
    G = np.zeros([2 * n, 2 * n])
    for i, j in XX:
        G[i, j] = 1
    for i, j in XR:
        G[i, j + n] = 1
    return G

def DAG2OracleANM(G):
    n=len(G)//2
    oracleG=G.copy()
    for i in range(n):
        if not is_anm_identifiable_basis(G,i):
            pa_i=parent(G,i)
            for p in pa_i:
                #oracleG[p,i]=1
                oracleG[i,p]=1
    return oracleG

@numba.njit
def is_anm_identifiable_basis(G, j):
    n = len(G) // 2
    pa_j = set(parent(G, j))
    # j父亲不存在双向边，看j的子孙是否存在pa_j或j的indicator

    des = set()
    des.add(j)
    des.pop()
    des = find_R_descendent_basis(G, j, des)
    for p in pa_j:
        if not is_bi_edge(G,p,j):
            if p + n in des or j + n in des:
                # 如果pa_j或j的indicator是des的子代则anm可能无法识别，因为有些双向边未必是真的是这个方向
                return False
    for p in pa_j:
        # 当我们选择p作为parent的时候，令p->j
        if is_bi_edge(G,p,j):
            des_p = set()
            des_p.add(j)
            des_p.pop()
            G[p, j] = 1
            G[j, p] = 0
            if not is_cycle(G,p,j,List.empty_list(types.int64)):
                des_p = find_R_descendent_basis(G, j,des_p)
                G[p, j] = 1
                G[j, p] = 1
                if p + n in des_p or j + n in des_p:
                    # 如果pa_j或j的indicator是des的子代则anm可能无法识别，因为有些双向边未必是真的是这个方向
                    return False

    # 如果考虑所有双向边的子代情况下都不违反anm，则必然可识别
    return True

@numba.njit
def find_R_descendent_basis(G,i,des):
    # 这个版本要实现dfs的时候不能形成环，且开始递归了
    n=len(G)//2
    if i>=n:
        des.add(i)
    for c in child(G, i):
        if c>=n:
            des.add(c)
        elif not is_bi_edge(G,i,c) and not is_cycle(G,i,c,List.empty_list(types.int64)):
            find_R_descendent_basis(G, c, des)
        else:
            G[i,c]=1
            G[c,i]=0
            if not is_cycle(G,i,c,List.empty_list(types.int64)) and not is_anm_identifiable_basis(G,c): # 保证在遍历descendent的时候不形成环
                find_R_descendent_basis(G, c,des)
            G[i, c] = 1
            G[c, i] = 1
    return des


def plotG(G):
    n=len(G)//2
    labels = []
    for i in range(n):
        labels.append("x"+str(i))
    for i in range(n):
        labels.append("R" + str(i))
    df = pd.DataFrame(G, index=labels, columns=labels)
    df=df.loc[(df.sum(0) != 0) | (df.sum(1) != 0), (df.sum(0) != 0) | (df.sum(1) != 0)]

    G2 = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    #pos = graphviz_layout(G2, prog="dot")

    pos = nx.nx_pydot.graphviz_layout(G2)
    #pos = nx.spring_layout(G2,iterations=20)

    #nx.draw_circular(G2, with_labels=True)
    nx.draw_networkx(G2,pos,with_labels=True)
    plt.axis('equal')
    plt.show()
    
