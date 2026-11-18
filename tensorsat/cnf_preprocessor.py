from typing import List, Tuple, Set, Dict

class CNFFormula:
    """Represents a Boolean formula in Conjunctive Normal Form (CNF)."""
    def __init__(self, n_vars: int, clauses: List[List[int]]):
        self.n_vars = n_vars
        self.clauses = clauses
        
    @classmethod
    def from_dimacs(cls, dimacs_str: str) -> "CNFFormula":
        """Parse DIMACS CNF format string."""
        clauses = []
        n_vars = 0
        for line in dimacs_str.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p cnf'):
                parts = line.split()
                n_vars = int(parts[2])
                continue
            lits = [int(x) for x in line.split() if int(x) != 0]
            if lits:
                clauses.append(lits)
        return cls(n_vars, clauses)

    def extract_variable_interaction_graph(self) -> Dict[int, Set[int]]:
        """
        Builds the primal variable interaction graph (edges between variables
        co-occurring in at least one clause).
        """
        adj: Dict[int, Set[int]] = {i: set() for i in range(1, self.n_vars + 1)}
        for clause in self.clauses:
            vars_in_clause = [abs(lit) for lit in clause]
            for i in range(len(vars_in_clause)):
                for j in range(i + 1, len(vars_in_clause)):
                    u, v = vars_in_clause[i], vars_in_clause[j]
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def partition_sparse_and_dense_cliques(self, density_threshold: float = 0.5) -> Tuple[List[Set[int]], List[List[int]]]:
        """
        Identifies dense sub-clusters (cliques) for tensor uncrossing,
        separating them from sparse clauses routed to standard CDCL.
        """
        adj = self.extract_variable_interaction_graph()
        dense_clusters: List[Set[int]] = []
        visited = set()
        
        for v in range(1, self.n_vars + 1):
            if v in visited:
                continue
            neighbors = adj[v]
            if not neighbors:
                continue
            # Check clique density
            k = len(neighbors) + 1
            if k >= 4:
                cluster = set(neighbors) | {v}
                # Check internal edge count
                internal_edges = 0
                for u in cluster:
                    internal_edges += len(adj[u] & cluster)
                max_edges = k * (k - 1)
                density = internal_edges / max_edges if max_edges > 0 else 0
                if density >= density_threshold:
                    dense_clusters.append(cluster)
                    visited.update(cluster)
                    
        # Clauses belonging strictly to sparse vs dense
        dense_vars = set().union(*dense_clusters) if dense_clusters else set()
        sparse_clauses = [
            c for c in self.clauses if not any(abs(l) in dense_vars for l in c)
        ]
        return dense_clusters, sparse_clauses
