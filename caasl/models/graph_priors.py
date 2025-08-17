import igraph as ig
import torch


class ErdosRenyi(object):
    """
    Erdos-Renyi random graph

    Args:
        num_nodes (int): number of nodes in the graph
        degree (float): expected number of edges per node, scaled to account for number of nodes
        n_parallel (int): number of parallel graphs to sample
    """

    def __init__(self, num_nodes, degree=0.5, n_parallel=1):
        self.edges_per_var = degree
        self.num_nodes = num_nodes
        n_edges = self.edges_per_var * num_nodes
        self.n_parallel = n_parallel
        p = min(n_edges / ((num_nodes * (num_nodes - 1)) / 2), 0.99)
        self.mat = torch.distributions.Bernoulli(probs=p)

    def __call__(self, num_theta=1):
        # select p s.t. we get requested edges_per_var in expectation

        # sample
        init_mat = self.mat.sample(
            (num_theta, self.n_parallel, 1, self.num_nodes, self.num_nodes)
        )

        # make DAG by zeroing above diagonal
        dag_l = torch.tril(init_mat, diagonal=-1)

        # randomly permute
        eye = torch.eye(self.num_nodes)
        p = eye[torch.randperm(self.num_nodes)]
        dag = p.T @ dag_l @ p
        return dag


class ScaleFree(object):
    """
    Barabasi-Albert (scale-free)
    Power-law in-degree

    Args:
       num_nodes (int): number of nodes in the graph
       degree (int): number of edges per node
       power (float): power in preferential attachment process.
            Higher values make few nodes have high in-degree.
       n_parallel (int): number of parallel graphs to sample

    """

    def __init__(self, num_nodes, degree=1, power=1.0, n_parallel=1):
        if type(degree) is float:
            degree = int(degree) + 1
        self.num_nodes = num_nodes
        self.edges_per_var = degree
        self.power = power
        self.n_parallel = n_parallel

    def __call__(self, num_theta=1):
        graphs_outer = []
        for _ in range(num_theta):
            graphs = []
            for _ in range(self.n_parallel):
                perm = torch.randperm(self.num_nodes).tolist()
                g = ig.Graph.Barabasi(
                    n=self.num_nodes,
                    m=self.edges_per_var,
                    directed=True,
                    power=self.power,
                ).permute_vertices(perm)
                g = torch.tensor(g.get_adjacency().data)
                graphs.append(g)
            graphs = torch.stack(graphs).to(torch.float32)
            graphs_outer.append(graphs)

        return torch.stack(graphs_outer).unsqueeze(-3)
