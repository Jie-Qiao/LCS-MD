import math
import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy
from itertools import combinations
from sklearn.linear_model import LinearRegression

from causallearn2.graph.GraphClass import CausalGraph
from causallearn2.utils.cit import fisherz
from causallearn2.utils.cit import CIT

from lib.ANM_rule_path import anm_rule_by_paths,DAG2OracleANM


# create missing mechanism(MV-SELF)
def randomMissGraph(dag, missingness='SMAR', rom=1 / 3, num_self_node=2, randomState=None):
    '''
    :param dag: true DAG from bnlearn
    :param B_true: binarrary_true
    :param varnames:varible names
    :param missingness: type of missing mechanism
    :param rom: ratio of missing variables
    :return: parent of missingness for each partially observed variable. {'13': ['10']} means X_10->R_13
    '''

    '''
    :param dag: true DAG
    :param missingness: type of missing mechanism
    :param rom: ratio of missing variables
    :return: parent of missingness for each partially observed variable
    '''
    cause_dict = {}
    if randomState is None:
        randomState=np.random.RandomState()
    if missingness == 'SMAR':
        num_vars_miss = round(len(dag.nodes) * rom)
        num_other_vars_miss = num_vars_miss - num_self_node
        candidate_other_vars_miss = []
        candidate_self_vars_miss = []

        # find self node
        #1 Conform to assumptions as much as possible
        for var in list(dag.nodes):
            neighbours = list(dag.adj[var])
            childrens = list(dag.successors(var))
            if len(childrens) > 0 and len(neighbours) == 2:
                for two_neighbour in combinations(neighbours, 2):
                    if nx.d_separated(dag,set([two_neighbour[0]]), set([two_neighbour[1]]), set([var])):
                        # 遍历graph的每个结点，如果该结点存在两个邻居，且被该结点d分离，则加入到候选self missing结点中。
                        candidate_self_vars_miss.append(var)

        if len(candidate_self_vars_miss) < num_self_node:
            # print("Cannot find enough self nodes")
            return None

        self_node = []
        for i in range(num_self_node):
            self_node.append(candidate_self_vars_miss[i])
        for var_miss in self_node:
            cause_dict[str(var_miss)] = [str(var_miss)] # 这是缺失的selfmasking变量，他的parent是他自己

        # find mar
        if num_other_vars_miss >= 1: # 如果还需要除了selfmasking以外的缺失变量
            vars_complete = list(sorted(set(dag.nodes).difference(set(self_node)))) # 剩余变量
            vars_miss = randomState.choice(vars_complete, num_other_vars_miss).tolist()          # Randomly select mar of variables as missing variables
            vars_complete = [v for v in vars_complete if v not in vars_miss] # 新挑的剩余变量
            for var in vars_miss: # 遍历R_i
                children = list(dag.successors(var)) # 找到V_i非self masking缺失变量中的孩子结点
                children = [v for v in list(children) if v in vars_complete]  # 找到V_i非self masking缺失变量中的非缺失孩子结点
                if randomState.uniform(0, 1) > 0.2: # 以0.2的概率设置为随机缺失，否则就是完全随机缺失
                    if len(children) > 0: # 如果V_i存在非缺失孩子，则以一定概率选择其中一个，将其设置为R_i的parent
                        cause_dict[str(var)] = [str(var) for var in randomState.choice(children, 1).tolist()]
                    else: # 如果V_i不存在非缺失孩子，则随机选择没缺失的变量，将其设置为R_i的parent
                        cause_dict[str(var)] = [str(var) for var in randomState.choice(vars_complete, 1).tolist()]
                else:
                    cause_dict[str(var)] = []
            return cause_dict
        else:
            return cause_dict

    elif missingness == 'SMNAR':
        num_vars_miss = round(len(dag.nodes) * rom)
        num_other_vars_miss = num_vars_miss - num_self_node
        candidate_self_vars_miss = []

        # find self node
        for var in list(dag.nodes):
            neighbours = list(dag.adj[var])
            childrens = list(dag.successors(var))
            if len(childrens) > 0 and len(neighbours) == 2:
                for two_neighbour in combinations(neighbours, 2):
                    if nx.d_separated(dag,set([two_neighbour[0]]), set([two_neighbour[1]]), set([var])):
                        candidate_self_vars_miss.append(var)


        if len(candidate_self_vars_miss) < num_self_node:
            # print("Cannot find enough self nodes")
            return None
        # find selfmasking
        self_node = []
        for i in range(num_self_node):
            self_node.append(candidate_self_vars_miss[i])
        for var_miss in self_node:
            cause_dict[str(var_miss)] = [str(var_miss)]

        # find mnar
        if num_other_vars_miss >= 1:
            vars_complete = list(sorted(set(dag.nodes).difference(set(self_node))))
            candidate_other_vars_miss = randomState.choice(vars_complete, num_other_vars_miss).tolist()
            vars_complete = list(sorted(set(vars_complete).difference(set(candidate_other_vars_miss))))
            for var in candidate_other_vars_miss:
                children = list(dag.successors(var)) # neighbour and Fully observable variable
                if randomState.uniform(0, 1) > 1: # 这是MAR的
                    children = [v for v in list(children) if v in vars_complete]
                    if len(children) > 0:
                        cause_dict[str(var)] = [str(var) for var in  randomState.choice(children, 1).tolist()]
                    else:
                        cause_dict[str(var)] = [str(var) for var in randomState.choice(vars_complete, 1).tolist()]
                else:
                    children = [v for v in list(children) if v in candidate_other_vars_miss]
                    #mnar node num > 2 , error
                    if len(children) > 0:
                        cause_dict[str(var)] = [str(var) for var in randomState.choice(children, 1).tolist()]
                    else:
                        # mnar node num = 1 and other all self node, ->mar
                        candidate_other_vars_miss2=candidate_other_vars_miss.copy().remove(var)
                        if len(candidate_other_vars_miss2) == 0:
                            cause_dict[str(var)] = [str(var) for var in randomState.choice(vars_complete, 1).tolist()]
                        else:
                            cause_dict[str(var)] = [str(var) for var in randomState.choice(candidate_other_vars_miss2, 1).tolist()]
            return cause_dict
        else:
            return cause_dict

    else:
        raise Exception('missingness ' + missingness + ' is undefined.')
    return

def randomMissGraph2(dag, missingness='SMAR', rom=1 / 3, num_self_node=2, randomState=None):

    if randomState is None:
        randomState=np.random.RandomState()
    if missingness == 'SMAR':
        return randomMARGraph(dag,rom,num_self_node,randomState)
    elif missingness == 'SMNAR':
        return randomMNARGraph(dag, rom, num_self_node, randomState)
    else:
        raise NotImplementedError


def randomMARGraph(dag, rom=1 / 3, num_self_node=2, randomState=None):
    cause_dict={}
    if randomState is None:
        randomState=np.random.RandomState()
    num_vars_miss = int(len(dag.nodes) * rom)
    num_other_vars_miss = num_vars_miss - num_self_node
    candidate_self_vars_miss=[]
    for var in list(dag.nodes):
        neighbours = list(dag.adj[var])
        for two_neighbour in combinations(neighbours, 2):
            if nx.algorithms.d_separation.is_d_separator(dag, set([two_neighbour[0]]), set([two_neighbour[1]]), set([var])):
                candidate_self_vars_miss.append(var)
                break

    if len(candidate_self_vars_miss) < num_self_node:
        # print("Cannot find enough self nodes")
        return None
    # find selfmasking
    self_node = []
    for i in range(num_self_node):
        self_node.append(candidate_self_vars_miss[i])
    for var_miss in self_node:
        cause_dict[str(var_miss)] = [str(var_miss)]
    if num_other_vars_miss>0:
        vars_complete = list(sorted(set(dag.nodes).difference(set(self_node))))  # 剩余变量
        vars_miss = randomState.choice(vars_complete,num_other_vars_miss).tolist()  # Randomly select mar of variables as missing variables
        vars_complete = [v for v in vars_complete if v not in vars_miss]  # 新挑的剩余变量

        for var in vars_miss:  # 遍历R_i
            neighbours = list(dag.adj[var])  # 找到V_i非self masking缺失变量中的孩子结点
            neighbours = [v for v in neighbours if v in vars_complete]  # 找到V_i非self masking缺失变量中的非缺失孩子结点
            if randomState.uniform(0, 1) > 0.2:  # 以0.2的概率设置为随机缺失，否则就是完全随机缺失
                if len(neighbours) > 0:  # 如果V_i存在非缺失孩子，则以一定概率选择其中一个，将其设置为R_i的parent
                    cause_dict[str(var)] = [str(var) for var in randomState.choice(neighbours, 1).tolist()]
                else:  # 如果V_i不存在非缺失孩子，则随机选择没缺失的变量，将其设置为R_i的parent
                    cause_dict[str(var)] = [str(var) for var in randomState.choice(vars_complete, 1).tolist()]
            else:
                cause_dict[str(var)] = []
    cause_dict=dict(sorted(cause_dict.items(), key=lambda x: int(x[0]))) # 让causa dict以确实变量index的顺序排列
    return cause_dict

def randomMNARGraph(dag, rom=1 / 3, num_self_node=2, randomState=None):
    cause_dict={}
    if randomState is None:
        randomState=np.random.RandomState()
    num_vars_miss = int(len(dag.nodes) * rom)
    num_other_vars_miss = num_vars_miss - num_self_node
    candidate_self_vars_miss=[]
    for var in list(dag.nodes):
        neighbours = list(dag.adj[var])
        for two_neighbour in combinations(neighbours, 2):
            if nx.algorithms.d_separation.is_d_separator(dag, set([two_neighbour[0]]), set([two_neighbour[1]]), set([var])):
                candidate_self_vars_miss.append(var)
                break

    if len(candidate_self_vars_miss) < num_self_node:
        # print("Cannot find enough self nodes")
        return None
    # find selfmasking
    self_node = []
    for i in range(num_self_node):
        self_node.append(candidate_self_vars_miss[i])
    for var_miss in self_node:
        cause_dict[str(var_miss)] = [str(var_miss)]
    if num_other_vars_miss > 0:
        vars_complete = list(sorted(set(dag.nodes).difference(set(self_node))))  # 剩余变量
        vars_miss = randomState.choice(vars_complete,num_other_vars_miss).tolist()  # Randomly select mar of variables as missing variables
        for var in vars_miss:  # 遍历R_i
            #neighbours = list(dag.adj[var])  # 找到V_i非self masking缺失变量中的孩子结点
            remain_nodes= [str(v) for v in dag.nodes if v not in self_node]
            cause_dict[str(var)] = randomState.choice(remain_nodes,1).tolist()
    cause_dict=dict(sorted(cause_dict.items(), key=lambda x: int(x[0]))) # 让causa dict以确实变量index的顺序排列
    return cause_dict


# add missing value in dataset
def add_missing(data, cause_dict, m_min=0.1, m_max=0.6,quantile=0.2,randomState=None):
    if randomState is None:
        randomState=np.random.RandomState()
    data_missing = deepcopy(data)
    for var in cause_dict.keys():
        if len(cause_dict[var]) == 0:
            m = randomState.uniform(m_min, m_max)
            data_missing[var][randomState.uniform(size=len(data)) < m] = np.nan  #default uniform(low=0,high=1)
        else:#Each item has m probability to fall in [0,m], the total interval [0,1]
            for cause in cause_dict[var]:
                # ser_up_down = regression_equation(data, var, cause)
                thres = data[cause].quantile(quantile)
                # data_missing[var][(np.random.uniform(size=len(data)) < 0.6) & (data[cause] < thres) & (ser_up_down < 0)] = np.nan
                # data_missing[var][(np.random.uniform(size=len(data)) < 0.1) & (data[cause] >= thres) & (ser_up_down > 0)] = np.nan

                data_missing[var][(randomState.uniform(size=len(data)) < m_max) & (data[cause] < thres)] = np.nan
                data_missing[var][(randomState.uniform(size=len(data)) < m_min) & (data[cause] >= thres)] = np.nan
    return data_missing

# Calculate the regression line for two variables
def regression_equation(data, X_i, X_Pi):

    # dataFrame 转 numpy
    X_i = data[X_i]
    X_Pi = data[X_Pi]

    X_i = X_i.values.reshape(X_i.shape[0], 1)
    X_Pi = X_Pi.values.reshape(X_Pi.shape[0], 1)
    X_Pi = X_Pi*X_Pi

    linreg = LinearRegression()
    reg = linreg.fit(X_Pi, X_i)
    predict_Xi = reg.predict(X_Pi)

    # plt.scatter(X_Pi, X_i, color='green')
    # plt.plot(X_Pi, reg.predict(X_Pi), color='red', linewidth=3)
    # plt.show()

    # With predict_ Xi is the dividing line, the top is positive and the bottom is negative
    up_down = X_i - predict_Xi

    # plt.scatter(X_Pi, X_i, color='green')
    # plt.plot(X_Pi, up_down, color='red', linewidth=3)
    # plt.show()

    up_down_list = map(lambda x: x[0], up_down)
    # to series
    ser_up_down = pd.Series(up_down_list)
    return ser_up_down



# new code
def power_with_original_sign(num_array, pow_num):
    size = num_array.shape[0]
    result = np.zeros(size)
    for i in range(size):
        if num_array[i] >= 0:
            result[i] = math.pow(num_array[i], pow_num)
        else:
            result[i] = - math.pow(abs(num_array[i]), pow_num)
    return result

# causal graph to adjacency_matrix
def cg2mDAG(cg, vars_miss=None, skeleton=False):
    G = deepcopy(cg.G.graph.T)
    n=len(G)
    for i in range(n):
        for j in range(n):
            if G[i][j] == -1 and G[j][i] == -1:
                G[i][j] = 1
                G[j][i] = 1
            elif cg.G.graph[i][j] == 1 and cg.G.graph[j][i] == -1:
                G[i][j] = 1
                G[i][j] = 0
    if skeleton:
        for i in range(n):
            for j in range(n):
                if G[i][j] == 1 and G[j][i] == 0:
                    G[i][j] = 1
                    G[j][i] = 1
    if vars_miss is not None:
        G = np.pad(G, ((0, len(vars_miss)), (0, len(vars_miss))), 'constant')
        R = np.zeros((n + len(vars_miss), n + len(vars_miss)))
        if len(cg.prt_m) > 0:
            # Assign learned R parent node and calumniate learned_selfmask_num
            for i in range(len(vars_miss)):
                R_i = int(vars_miss[i])
                if R_i in cg.prt_m['m']:
                    R_in_prt_m_index = cg.prt_m['m'].index(R_i)
                    Pas_R = cg.prt_m['prt'][R_in_prt_m_index].tolist()
                    for Pa_R in Pas_R:
                        R[Pa_R][n + i] = 1

        G = G + R

    return G


def cg2FullmDAG(cg, vars_miss):
    adj=cg.G.graph.T.copy()
    B_DAG = np.pad(adj, ((0, adj.shape[0]), (0, adj.shape[1])), 'constant')
    R_learned = np.zeros((adj.shape[0]*2, adj.shape[1] *2), dtype=int)

    if len(cg.prt_m) > 0:
        # Assign learned R parent node and calumniate learned_selfmask_num
        for i in range(len(vars_miss)):
            R = int(vars_miss[i])
            if R in cg.prt_m['m']:
                R_in_prt_m_index = cg.prt_m['m'].index(R)
                Pa_R = cg.prt_m['prt'][R_in_prt_m_index].tolist()
                for p in Pa_R:
                    # R_learned[Pa_R][adj.shape[1] + i] = 1
                    R_learned[p][adj.shape[1] + R] = 1

    FullmDAG = B_DAG + R_learned
    for i in range(FullmDAG.shape[0]):
        for j in range(FullmDAG.shape[1]):
            if FullmDAG[i][j] == -1 and FullmDAG[j][i] == -1:
                FullmDAG[i][j] = 1
                FullmDAG[j][i] = 1
            elif FullmDAG[i][j] == 1 and FullmDAG[j][i] == -1:
                FullmDAG[i][j] = 1
                FullmDAG[j][i] = 0
    return FullmDAG


def DAG2ANME(G, cause_dict):
    dim=len(G)
    G = np.pad(G, ((0, dim), (0, dim)), 'constant')
    R = np.zeros((dim * 2 , dim * 2), dtype= int)

    for R_i in cause_dict.keys():
        for Pa_R in cause_dict[R_i]:
            R[int(Pa_R)][int(R_i) + dim] = 1
    G = G + R

    G = DAG2OracleANM(G)
    G2, paths = anm_rule_by_paths(G.copy())

    indexSelect = list(range(dim))
    for R_i in cause_dict.keys():
        R_index = dim + int(R_i)
        indexSelect.append(R_index)
    B_R_DAG = G2[np.ix_(indexSelect, indexSelect)]

    return B_R_DAG

def DAG2mDAG(G, cause_dict: dict):
    "这个方法是将DAG转换padding后的图，然后再把多余的R（列为空）删除掉"
    dim=len(G)
    G = np.pad(G, ((0, dim), (0, dim)), 'constant')
    R = np.zeros((dim * 2 , dim * 2), dtype= int)

    for R_i in cause_dict.keys():
        for Pa_R in cause_dict[R_i]:
            R[int(Pa_R)][int(R_i) + dim] = 1
    G = G + R

    indexSelect = list(range(dim))
    for R_i in cause_dict.keys():
        R_index = dim + int(R_i)
        indexSelect.append(R_index)
    mDAG = G[np.ix_(indexSelect, indexSelect)]
    return mDAG

def DAG2FullmDAG(G,cause_dict):
    dim=len(G)
    G = np.pad(G, ((0, dim), (0, dim)), 'constant')
    R = np.zeros((dim * 2 , dim * 2), dtype= int)

    for R_i in cause_dict.keys():
        for Pa_R in cause_dict[R_i]:
            R[int(Pa_R)][int(R_i) + dim] = 1
    G = G + R
    return G

def mDAG2FullmDAG(G,varmiss):
    varmiss=[int(x) for x in varmiss]
    varmiss=sorted(varmiss)
    dim=len(G)-len(varmiss)
    G=G.copy()
    mDAG=G[0:dim,0:dim]
    mDAG = np.pad(mDAG, ((0, dim), (0, dim)), 'constant')
    R = np.zeros((dim * 2, dim * 2), dtype=int)
    mDAG = mDAG + R
    for i,R_i in enumerate(varmiss):
        mDAG[0:dim,dim+R_i]=G[0:dim,dim+i]
        #mDAG[dim + R_i, :] = G[dim + i,:]
    return mDAG



def simulation_to_generate_cg(data,B_R_DAG, varnames):

    # X--Y -1-- -1 to  X--Y  1--1
    B_true_learn = deepcopy(B_R_DAG)
    for i in range(B_R_DAG.shape[0]):
        for j in range(B_R_DAG.shape[1]):
            if B_R_DAG[i][j] == -1 and B_R_DAG[j][i] == -1:
                B_true_learn[i][j] = 1
                B_true_learn[j][i] = 1
            elif B_R_DAG[i][j] == 1 and B_R_DAG[j][i] == -1:
                B_true_learn[i][j] = 1
                B_true_learn[j][i] = 0

    B_true_learn = B_true_learn[:len(varnames), :len(varnames)]


    verbose = True
    #True iff verbose output should be printed.
    indep_test = fisherz
    indep_test = CIT(data, indep_test)

    no_of_var = len(varnames)
    node_names = varnames
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)

    return cg


def calculate_selfmasking_rate(mDAG, cause_dict: dict):
    dim= len(mDAG) - len(cause_dict.keys())
    learned_num_self_node = 0
    num_self_node=0
    for i,k in enumerate(cause_dict.keys()):
        if k in cause_dict[k]:
            num_self_node+=1
            if mDAG[int(k)][dim + i]:
                learned_num_self_node+=1
    return learned_num_self_node/num_self_node


def countSHD(B_R_learned, B_true_nd):
    shd = 0
    for i in range(B_R_learned.shape[0]):
        for j in range(i+1):
            # missing
            if (B_true_nd[i][j] == 1 or B_true_nd[j][i] == 1)&(B_R_learned[i][j] == 0 and B_R_learned[j][i] == 0) :
                shd = shd +1
            # extra
            if (B_true_nd[i][j] == 0 and B_true_nd[j][i] == 0)&(B_R_learned[i][j] == 1 or B_R_learned[j][i] == 1) :
                shd = shd + 1
            # incorrectly oriented
            if (B_true_nd[i][j] == 1 or B_true_nd[j][i] == 1)&(B_R_learned[i][j] == 1 or B_R_learned[j][i] == 1) :
                # reversed
                if (B_true_nd[i][j] == 1 and B_true_nd[j][i] == 0)&(B_R_learned[i][j] == 0 and B_R_learned[j][i] == 1):
                    shd = shd + 1
                if (B_true_nd[i][j] == 0 and B_true_nd[j][i] == 1)&(B_R_learned[i][j] == 1 and B_R_learned[j][i] == 0):
                    shd = shd + 1
                #undirected ----direct
                if (B_true_nd[i][j] == 1 and B_true_nd[j][i] == 1)& ((B_R_learned[i][j] == 0 and B_R_learned[j][i] == 1) or (B_R_learned[i][j] == 1 and B_R_learned[j][i] == 0)):
                    shd = shd + 1
                if ((B_true_nd[i][j] == 0 and B_true_nd[j][i] == 1) or (B_true_nd[i][j] == 1 and B_true_nd[j][i] == 0)) & (B_R_learned[i][j] == 1 and B_R_learned[j][i] == 1):
                    shd = shd + 1

    return shd

def F1(pred,real):
    TP=0
    FP=0
    TN=0
    FN=0
    #n=0
    nrow=len(real)
    ncol=len(real[0])
    for i in range(nrow):
        for j in range(ncol):
            if real[i,j]==1 and real[j,i]==0 and pred[i,j]==1 and pred[j,i]==0:
                TP=TP+1
                #n=n+1
            # if real[i,j]==1&real[j,i]==1&pred[i,j]==1&pred[j,i]==1:
            #     TP = TP + 1
            #     # n=n+1
            if real[i,j]==0 and real[j,i]==0 and pred[i,j]==1 and pred[j,i]==0:
                FP=FP+1
                #n=n+1
            if real[i,j]==0 and real[j,i]==0 and pred[i,j]==0 and pred[j,i]==0:
                TN=TN+1
                #n=n+1
            if real[i,j]==1 and real[j,i]==0 and pred[i,j]==0 and pred[j,i]==0:
                FN=FN+1
                #n=n+1
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    if precision+recall==0:
        f1=0
    else:
        f1=2*precision*recall/(precision+recall)
    return f1,precision,recall



