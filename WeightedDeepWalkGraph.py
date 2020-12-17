from csrgraph.graph import csrgraph
from csrgraph.methods import _update_src_array
from csrgraph.random_walks import _random_walk
import numpy as np

import numba
from numba import jit, jitclass
from scipy import sparse


@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def _damp_and_row_norm(weights, src):
    """
    Returns the weights for normalized rows in a CSR Matrix.

    Parameters
    ----------
    weights : array[float]
        The data array from a CSR Matrix.
        For a scipy.csr_matrix, is accessed by M.data

    src : array[int]
        The index pointer array from a CSR Matrix.
        For a scipy.csr_matrix, is accessed by M.indptr

    ----------------
    returns : array[float32]
        The normalized data array for the CSR Matrix
    """

    #damping part
    weights = np.maximum(weights, 1.1)
    weights = np.maximum(1.0/np.log(weights), np.e)

    #normalization
    n_nodes = src.size - 1
    res = np.empty(weights.size, dtype=np.float32)
    for i in numba.prange(n_nodes):
        s1 = src[i]
        s2 = src[i + 1]
        rowsum = np.sum(weights[s1:s2])
        res[s1:s2] = weights[s1:s2] / rowsum
    return res




class WeightedDeepWalkGraph(csrgraph):
    def __init__(self, data, nodenames=None, copy=True, threads=0):
        csrgraph.__init__(self, data, nodenames, copy, threads)

    def random_walks(self,
                     walklen=10,
                     epochs=1,
                     start_nodes=None,
                     normalize_self=False,
                     return_weight=1.,
                     neighbor_weight=1.):
        """
        Create random walks from the transition matrix of a graph
            in CSR sparse format
        Parameters
        ----------
        T : scipy.sparse.csr matrix
            Graph transition matrix in CSR sparse format
        walklen : int
            length of the random walks
        epochs : int
            number of times to start a walk from each nodes
        return_weight : float in (0, inf]
            Weight on the probability of returning to node coming from
            Having this higher tends the walks to be
            more like a Breadth-First Search.
            Having this very high  (> 2) makes search very local.
            Equal to the inverse of p in the Node2Vec paper.
        explore_weight : float in (0, inf]
            Weight on the probability of visitng a neighbor node
            to the one we're coming from in the random walk
            Having this higher tends the walks to be
            more like a Depth-First Search.
            Having this very high makes search more outward.
            Having this very low makes search very local.
            Equal to the inverse of q in the Node2Vec paper.
        threads : int
            number of threads to use.  0 is full use
        Returns
        -------
        out : 2d np.array (n_walks, walklen)
            A matrix where each row is a random walk,
            and each entry is the ID of the node
        """


        # Make csr graph
        if normalize_self:
            self.damp_and_normalize(return_self=True)
            T = self
        else:
            T = self.damp_and_normalize(return_self=False)
        n_rows = T.nnodes
        if start_nodes is None:
            start_nodes = np.arange(n_rows)
        sampling_nodes = np.tile(start_nodes, epochs)
        # Node2Vec Biased walks if parameters specified

        walks = _random_walk(T.weights, T.src, T.dst,
                                 sampling_nodes, walklen)
        return walks

    def damp_and_normalize(self, return_self=True):
        """
        Normalizes edge weights per node
        For any node in the Graph, the new edges' weights will sum to 1
        return_self : bool
            whether to change the graph's values and return itself
            this lets us call `G.normalize()` directly
        """
        new_weights = _damp_and_row_norm(self.weights, self.src)
        if return_self:
            self.weights = new_weights
            if hasattr(self, 'mat'):
                self.mat = sparse.csr_matrix((self.weights, self.dst, self.src))
            return self
        else:
            return csrgraph(sparse.csr_matrix(
                (new_weights, self.dst, self.src)),
                nodenames=self.names)


def _edgelist_to_wdw_graph(elist, nnodes, nodenames=None):
    """
    Assumptions:
        1) edgelist is sorted by source nodes
        2) nodes are all ints in [0, num_nodes]
    Params:
    ---------
    elist : pd.Dataframe[src, dst, (weight)]
        df of edge pairs. Assumed to be sorted.
        If w weight column is present, named 'weight'

    Return:
    ----------
    csrgraph object

    """
    dst = elist.dst.to_numpy()
    src = np.zeros(nnodes + 1)
    # Now fill indptr array
    src[0] = 0 # each idx points to node start idx

    # Use a groupby -> maxvalue to fill indptr
    elist['cnt'] = np.ones(elist.shape[0])
    grp = (elist[['cnt', 'src']]
        # Max idx per node
        .groupby('src')
        .count()
        .reset_index(drop=False)
    )
    _update_src_array(src, grp.src.to_numpy(), grp.cnt.to_numpy())
    elist.drop(columns=['cnt'], inplace=True)
    if 'weight' in elist.columns:
        weights = elist[elist.columns[-1]].astype(np.float)
    else:
        weights = np.ones(dst.shape[0])
    return WeightedDeepWalkGraph(
        sparse.csr_matrix((weights, dst, src)),
        nodenames=nodenames
    )
