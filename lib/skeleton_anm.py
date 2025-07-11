from copy import deepcopy

from causallearn2.search.ConstraintBased.PC import pc


from lib.utils import cg2mDAG
from lib.LCS_MD import lcs_md, SM_MVPC



from utils.utils import postprocess


def TD_PC(mdata, alpha, mv_fisherz, skeleton= True, node_names=None):
    cg_mar = pc(mdata, alpha, mv_fisherz, True, 0, -1,skeleton=skeleton, node_names=node_names)
    return cg_mar

def MVPC(mdata, alpha, mv_fisherz, skeleton= True,node_names=None):
    mvpc_cg_mar = pc(mdata, alpha, mv_fisherz, True, 0, -1,True, skeleton=skeleton,node_names=node_names)

    return mvpc_cg_mar

def skeleton_anm(data, alpha = 0.05,alpha2 = 0.05, method='LCS-MD', datatype='linear', skeleton=True,priority=0 ):
    '''

    '''

    dc = data.nunique() == 1
    dc = dc[dc].index.values
    data = data.drop(dc, axis = 1)  #Drop the column that only contains 1 value (all NaN).

    varnames = data.columns.tolist()
    vars_miss = data.columns[data.isnull().any()].tolist()
    node_names = varnames
    data = data.to_numpy()
    mdata = data.copy()
    if datatype=="linear":
        independent_test="mv_fisherz"
    else:
        independent_test = "mv_fisherz"

    #independent_test="hsic_spectral"
    #independent_test="hsic_block"
    if method == 'SM-MVPC':
        #mvpc_cg_mar = SM_MVPC(mdata, alpha, independent_test, True,node_names,False)
        # mvpc_cg_mar = SM_MVPC(mdata, alpha, "kci", True,node_names,False)

        mvpc_cg_mar = SM_MVPC(mdata, alpha,independent_test, True,node_names,False)

        mDAG = cg2mDAG(mvpc_cg_mar, vars_miss, skeleton=skeleton)

    elif method == 'LCS-MD':
        mDAG = lcs_md(data, independent_test, varnames, vars_miss, alpha, alpha2, datatype = datatype,priority=priority)
    elif method == 'SM-MVPC-ANM':
        mDAG = lcs_md(data, independent_test, varnames, vars_miss, alpha, alpha2, datatype = datatype, apply_rule=False,priority=priority)
    elif method == 'TD-PC':
        cg_mar = TD_PC(mdata, alpha, "mv_fisherz", skeleton= skeleton, node_names=node_names)  # Run PC and obtain the estimated graph (CausalGraph object)
        mDAG = cg2mDAG(cg_mar, vars_miss, skeleton=skeleton)
    elif method == 'MVPC':
        mvpc_cg_mar = MVPC(mdata, alpha, "mv_fisherz", skeleton=skeleton, node_names=node_names)  # Run PC and obtain the estimated graph (CausalGraph object
        mDAG = cg2mDAG(mvpc_cg_mar, vars_miss, skeleton=skeleton)
    else:
        raise Exception('The input method: ' + method + ' is invalid.')

    return mDAG