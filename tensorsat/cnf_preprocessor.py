from typing import List, Tuple, Set, Dict

class CNFFormula:
    """Represents a Boolean formula in Conjunctive Normal Form (CNF)."""
    def __init__(self, n_vars: int, clauses: List[List[int]]):
        self.n_vars = n_vars
        self.clauses = clauses
        
    @classmethod
    def from_dimacs(cls, dimacs_str: str, strict: bool = False) -> "CNFFormula":
        """
        Parse DIMACS CNF format string by token streaming.
        Clauses are strictly delimited by terminal 0 tokens.
        Empty clauses (a 0 immediately following a 0 or at start) are preserved.

        Args:
            dimacs_str: DIMACS format string.
            strict: When True, enforces valid header, non-negative declared counts,
                    bounds checking for variable IDs (1 <= var <= n_vars), and
                    checks that all non-empty clauses end with terminal 0.
        """
        clauses: List[List[int]] = []
        n_vars = 0
        n_clauses_decl = None
        has_header = False
        current_clause: List[int] = []

        # Process token by token across lines
        for line_num, line in enumerate(dimacs_str.splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p cnf'):
                if has_header and strict:
                    raise ValueError(f"Duplicate 'p cnf' header at line {line_num}")
                parts = line.split()
                if len(parts) < 4:
                    if strict:
                        raise ValueError(f"Malformed 'p cnf' header at line {line_num}: expected 4 tokens")
                    n_vars = int(parts[2]) if len(parts) >= 3 else 0
                else:
                    n_vars = int(parts[2])
                    n_clauses_decl = int(parts[3])
                    if strict and (n_vars < 0 or n_clauses_decl < 0):
                        raise ValueError(f"Declared negative variable/clause count in header at line {line_num}")
                has_header = True
                continue

            # Tokenize by whitespace
            tokens = line.split()
            for token in tokens:
                try:
                    lit = int(token)
                except ValueError:
                    if strict:
                        raise ValueError(f"Non-integer literal '{token}' at line {line_num}")
                    continue

                if lit == 0:
                    clauses.append(current_clause)
                    current_clause = []
                else:
                    var = abs(lit)
                    if strict and n_vars > 0 and (var < 1 or var > n_vars):
                        raise ValueError(
                            f"Variable {var} out of declared bounds [1, {n_vars}] at line {line_num}"
                        )
                    current_clause.append(lit)

        # In case a trailing non-empty clause lacked a terminal 0
        if current_clause:
            if strict:
                raise ValueError("Unterminated trailing clause: missing final 0 delimiter")
            clauses.append(current_clause)

        if strict and n_clauses_decl is not None and len(clauses) != n_clauses_decl:
            raise ValueError(
                f"Clause count mismatch: declared {n_clauses_decl} in header, but parsed {len(clauses)}"
            )

        return cls(n_vars, clauses)

    def extract_variable_interaction_graph(self) -> Dict[int, Set[int]]:
        """
        Builds the primal variable interaction graph (edges between distinct variables
        co-occurring in at least one clause).
        Deduplicates variables per clause to prevent self-loops and false clique inflation.
        """
        adj: Dict[int, Set[int]] = {i: set() for i in range(1, self.n_vars + 1)}
        for clause in self.clauses:
            # Deduplicate and filter out any variables exceeding self.n_vars if non-strict
            vars_in_clause = sorted({abs(lit) for lit in clause if 1 <= abs(lit) <= self.n_vars})
            for i in range(len(vars_in_clause)):
                for j in range(i + 1, len(vars_in_clause)):
                    u, v = vars_in_clause[i], vars_in_clause[j]
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def partition_sparse_and_dense_cliques(
        self, density_threshold: float = 0.5
    ) -> Tuple[List[Set[int]], List[List[int]], List[List[int]], List[List[int]]]:
        """
        Identifies dense sub-clusters (cliques) for tensor uncrossing,
        and completely partitions the clauses into:
        1. dense_clusters: variable sets for each identified dense sub-graph.
        2. sparse_clauses: clauses containing only variables outside dense clusters.
        3. dense_clauses: clauses whose variables are entirely contained within a dense cluster.
        4. boundary_clauses: bridge clauses intersecting both dense and sparse variables (or multiple clusters).

        Preserves 100% of clause ownership across partition boundaries:
        len(sparse_clauses) + len(dense_clauses) + len(boundary_clauses) == len(self.clauses).
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

        dense_vars = set().union(*dense_clusters) if dense_clusters else set()

        sparse_clauses: List[List[int]] = []
        dense_clauses: List[List[int]] = []
        boundary_clauses: List[List[int]] = []

        for c in self.clauses:
            clause_vars = {abs(l) for l in c}
            if not clause_vars.intersection(dense_vars):
                # Purely outside dense clusters
                sparse_clauses.append(c)
            elif any(clause_vars.issubset(cluster) for cluster in dense_clusters):
                # Purely inside a single dense cluster
                dense_clauses.append(c)
            else:
                # Intersects dense and non-dense or spans multiple clusters
                boundary_clauses.append(c)

        return dense_clusters, sparse_clauses, dense_clauses, boundary_clauses
