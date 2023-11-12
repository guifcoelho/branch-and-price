# An example branch-and-price algorithm for the bin packing problem
#
from collections import defaultdict
from operator import itemgetter
import highspy, random, time, math
from itertools import combinations

# Relative gap for column generation
CG_GAP_REL = 1e-6

# Random seeds for reproducibility
SEED = 100
random.seed(SEED)


# Instance parameters
NumberItems = 150
ItemWeights = [round(random.uniform(1, 10), 1) for _ in range(NumberItems)]
BinCapacity = 15


#
# Solve instance with greedy first fit decreasing (FFD) heuristic
#
def solveGreedyModel():
    bins = defaultdict(float)
    solution = defaultdict(list)

    for item, w in sorted(enumerate(ItemWeights), reverse=True, key=itemgetter(1)):
        index = next((i for i, W in bins.items() if W + w <= BinCapacity), len(bins))
        bins[index] += w
        solution[index].append(item)
    
    return list(solution.values())

#
# The compact bin-packing model (for comparison with branch-and-price)
#
# min sum_{j} y_{j}
# s.t. 
#   sum_{j} x_{ij} == 1, \forall i                 // Each item i is assigned to exactly one bin
#   sum_{i} w_{i} * x_{ij} <= c * y_{j}, \forall j // Bin capacity constraint
#
#   x_{ij} = 1 if item i is assigned to bin j, 0 otherwise
#   y_{j}  = 1 if bin j is open, 0 otherwise
#
def solveCompactModel():
    # Estimate maximum number of bins with first fit decreasing (for smaller sized model)
    greedy_bins = solveGreedyModel()
    B = len(greedy_bins)

    m = highspy.Highs()
    m.setOptionValue('output_flag', True)
    m.setOptionValue('mip_abs_gap', 1-1e-5) # Since all variables in the objective are binary, it should stop when the absolute gap is below 1
    m.setOptionValue('random_seed', SEED)

    # add binary variables x_{ij} and y_{j}
    x = {(i,j): i*B + j for i in range(NumberItems) for j in range(B) }
    y = [len(x) + j for j in range(B)]

    m.addVars(len(x), [0]*len(x), [1]*len(x));  # x_{ij} \in {0,1}, \forall i,j
    m.addVars(len(y), [0]*len(y), [1]*len(y));  # y_{j}  \in {0,1}, \forall j
    m.changeColsIntegrality(
        len(x)+len(y),
        list(range(len(x)+len(y))),
        [highspy.HighsVarType.kInteger]*(len(x)+len(y))
    )

    # min sum_{j} y_{j}
    m.changeObjectiveSense(highspy.ObjSense.kMinimize)
    m.changeColsCost(len(y), y, [1]*len(y))
    
    #      \sum_{j} x_{ij} == 1, \foreach i 
    # 1 <= \sum_{j} x_{ij} <= 1
    for i in range(NumberItems):
        m.addRow(
            1,                                              # lhs
            1,                                              # rhs
            B,                                              # Number of non-zero variables
            [x[i,j] for j in range(B)],                     # Indexes of variable
            [1] * B                                         # Coefficients of variables
        )

    #         sum_{i} w_{i} * x_{ij} <= c * y_{j}, \foreach j
    # -inf <= sum_{i} w_{i} * x_{ij}  - c * y_{j} <= 0
    for j in range(B):
        m.addRow(
            -highspy.kHighsInf,                                             # lhs
            0,                                                              # rhs
            NumberItems+1,                                                  # Number of non-zero variables 
            [x[i,j] for i in range(NumberItems)] + [y[j]],                  # Indexes of variable
            [ItemWeights[i] for i in range(NumberItems)] + [-BinCapacity]   # Coefficients of variables  
        )
        
    m.run()
    vals = list(m.getSolution().col_value)

    bins = [
        [i for i in range(NumberItems) if vals[x[i,j]] > 0.9]
        for j in range(B)
        if vals[y[j]] > 0.9
    ]
    
    return bins

#
# Master problem for column generation
# A column represents a set of items packed into a bin
#
# min \sum_{k} \lambda_{k}
# s.t. 
#   \sum_{k \in K_i} \lambda_{k} == 1, \foreach i  (\mu_i duals)  // Each item i is packed exactly once (but may be spread over multiple columns)
#                                                                 //   where K_i is the set of columns where i appears
#   0 <= \lambda_{k} <= 1, \foreach k                             // Can use each column at most once (can be fractional)
#
def createMasterProblem(columns: list):
    m = highspy.Highs()
    m.setOptionValue('output_flag', False)
    m.setOptionValue('random_seed', SEED)
    m.setOptionValue('solver', 'ipm')
    
    m.addVars(len(columns), [0]*len(columns), [1]*len(columns))
    
    # min \sum_{k} \lambda_{k}
    m.changeObjectiveSense(highspy.ObjSense.kMinimize)
    m.changeColsCost(len(columns), list(range(len(columns))), [1]*len(columns))
    
    #      \sum_{k \in K_i} \lambda_{k} == 1, \foreach i
    # 1 <= \sum_{k \in K_i} \lambda_{k} <= 1
    for i in range(NumberItems):
        K_i = [k for k, column in enumerate(columns) if i in column]
        m.addRow(1, 1, len(K_i), K_i, [1]*len(K_i))

    m.run() 

    solution = m.getSolution()
    vals = list(solution.col_value)
    duals = list(solution.row_dual)

    return m, vals, duals

#
# Solve the knapsack subproblem exactly with HiGHS 
#
# 1 - max \sum_{i} \mu_{i} * z_{i}
# s.t.
#   \sum_{i} w_{i} * z_{i} <= capacity
#
#   z_{i} = 1 if item i is packed, 0 otherwise
#
def solveSubproblemExact(duals, branching_rule: list[tuple[tuple[int, int], float, int]] = None):
    assert len(duals) == NumberItems
    sp = highspy.Highs()
    sp.setOptionValue('output_flag', False)
    sp.setOptionValue('random_seed', SEED)

    sp.addVars(
        NumberItems,
        [0]*NumberItems,
        # [1 if dual >=0 else 0 for dual in duals]
        [1]*NumberItems
    )
    sp.changeColsIntegrality(NumberItems, list(range(NumberItems)), [highspy.HighsVarType.kInteger]*NumberItems)

    # max \sum_{i} \mu_{i} * z_{i}
    # where \mu_{i} is the dual variable for the i-th row of the master problem
    sp.changeColsCost(NumberItems, list(range(NumberItems)), duals)
    sp.changeObjectiveSense(highspy.ObjSense.kMaximize)

    #         sum_{i} w_{i} * z_{i} <= capacity, \foreach j
    # -inf <= sum_{i} w_{i} * z_{i} <= capacity
    sp.addRow(
        -highspy.kHighsInf,
        BinCapacity,
        NumberItems,
        list(range(NumberItems)),
        ItemWeights
    )
    if branching_rule is not None:
        for (r, s), _, rule in branching_rule:
            if rule == 1:
                # x_r = x_s
                sp.addRow(0, 0, 2, [r,s], [1,-1])
            else:
                #       x_r + x_s <= 1
                # -inf <= x_r + x_s <= 1
                sp.addRow(-highspy.kHighsInf, 1, 2, [r,s], [1,1])
    
    sp.run()

    if sp.getModelStatus() == highspy.HighsModelStatus.kOptimal:
        vals = list(sp.getSolution().col_value)
        new_column = sorted([i for i in range(NumberItems) if vals[i] > 0.9])

        return 1 - sp.getObjectiveValue(), new_column
    
    raise Exception("Subproblem is infeasible")


# Solve the knapsack subproblem with greedy heuristic
def solveSubproblemNotExact(duals, branching_rule: list[tuple[tuple[int, int], float, int]] = None):
    assert len(duals) == NumberItems
    total_weight = 0
    new_column = []
    
    new_column = []
    if branching_rule is not None:
        # Adds all pair with branching rule equals to 1
        for pair, _, rule in branching_rule:
            r, s = pair
            if rule == 1:
                sum_weigths = sum(ItemWeights[item] for item in pair)
                if total_weight + sum_weigths > BinCapacity and (r in new_column or s in new_column):
                    continue
                for item in pair:
                    if item not in new_column:
                        total_weight += ItemWeights[item]
                        new_column += [item]
        

        # Checks if there are any conflicts between rules.
        # If yes, then returns None (subproblem is infeasible)
        if len(new_column) > 0:
            disjoint_rows = [pair for pair, _, rule in branching_rule if rule == 0]
            for i1, i2 in combinations(new_column, 2):
                if (i1,i2) in disjoint_rows or (i2, i1) in disjoint_rows:
                    # for pair, rule in branching_rule:
                        # print(pair, rule)
                        # input()
                    raise Exception("Subproblem is infeasible")
            #         new_column.remove(r)
            #         new_column.remove(s)
            #         total_weight -= ItemWeights[r]
            #         total_weight -= ItemWeights[s]
            #         # raise Exception("Subproblem is infeasible")
        
            # if len(new_column) > 0:
            #     for (r,s), rule in branching_rule:
            #         if rule == 1 and ((r in new_column and s not in new_column) or (r not in new_column and s in new_column)):
            #             raise Exception("Subproblem is infeasible")


        # If the total weight of the required itens is greater than the capacity,
        # then returns None (subproblem is infeasible)
        if total_weight > BinCapacity:
            # input()
            raise Exception("Subproblem is infeasible (bin capacity)")

    for i in sorted(range(NumberItems), key=lambda i: -duals[i]/ItemWeights[i]):
        if i not in new_column:
            if duals[i] >= 0 and ItemWeights[i] + total_weight <= BinCapacity:

                # Adds new item to column only if there is not conflicts between rules
                can_add_item = True
                if branching_rule is not None:
                    for (r,s), _, rule in branching_rule:
                        if rule == 0 and ((i == r and s in new_column) or (i == s and r in new_column)):
                            can_add_item = False
                
                if can_add_item:
                    total_weight += ItemWeights[i]
                    new_column += [i]
    
    assert total_weight <= BinCapacity

    if len(new_column) == 0:
        raise Exception("Subproblem is infeasible")

    return 1 - sum(duals[i] for i in new_column), sorted(new_column)

def getFeasibleColumns(columns: list[list[int]], branching_rule: list[tuple[tuple[int, int], float, int]]):
    infeasible_columns = set()
    for (r, s), _, rule in branching_rule:
        for idx, column in enumerate(columns):
            if (
                (r in column or s in column)
                and (
                    (rule == 1 and (r not in column or s not in column))
                    or (rule == 0 and (r in column and s in column))
                )
            ):
                infeasible_columns.add(idx)

    return {idx for idx in range(len(columns)) if idx not in infeasible_columns}


#
# Generate columns for the master problem
#
def generateColumns(columns: list, mp: highspy.Highs, msg=True, solve_exact = False, start_time = 0, branching_rule:list[tuple[tuple[int, int], float, int]] = None):
    columns_ = columns.copy()
    best_gap = math.inf

    iter = 0
    while True:
        solution = mp.getSolution()
        assert len(list(solution.col_value)) == len(columns_), f"{iter} {solve_exact} - {len(list(solution.col_value))} != {len(columns_)}"
        duals = list(solution.row_dual)
        ub = mp.getObjectiveValue()

        # solve sub problem to generate new column
        try:
            if not solve_exact:
                obj_sub, new_column = solveSubproblemNotExact(duals[:NumberItems], branching_rule)
            else:
                obj_sub, new_column = solveSubproblemExact(duals[:NumberItems], branching_rule)
        except:
            return m, columns_, list(solution.col_value)

        if branching_rule is not None:
            obj_sub -= duals[-1]

        iter += 1     
        obj = ub + min(0, obj_sub)  # we terminate if obj_sub >= 0
        gap = (ub-obj)/ub
        if gap < best_gap:
            best_gap = gap
            if msg:
                row = [
                    f"{len(columns_)}",
                    f"{round(obj_sub, 3) :.4f}",
                    f'{round(obj, 3) :.4f}',
                    f"{round(ub, 3) :.4f}",
                    f"{max(0, gap) :.3%}", 
                    f"{round(time.perf_counter()-start_time, 2) :.2f}"
                ]
                if iter == 1:
                    header = ["Columns", 'Pricing', 'Obj', 'UB', 'gap', 'Time']
                    row_format = "".join([
                        "{:>"+str(max(len(row_el), len(header_el))+3)+"}"
                        for row_el, header_el in zip(row, header)
                    ])
                    print(row_format.format(*header))

                print(row_format.format(*row))

        # terminate if no column with good reduced cost is found
        # or if the subproblem is infeasible
        if obj_sub >= 0 or gap < CG_GAP_REL:# or len(new_column) == 0:
            # break
            num_cols = mp.getNumCol()
            assert num_cols == len(columns_), f"{iter} {solve_exact} - {num_cols} != {len(columns_)}"
            return m, columns_, list(solution.col_value)

        # Adds a new column to the master problem
        columns_ += [new_column]
        rows = new_column.copy()
        if branching_rule is not None:
            rows += [NumberItems]

        mp.addCol(
            1,                      # cost
            0,                      # lower bound
            1,                      # upper bound
            len(rows),              # number of rows
            rows,                   # indexes of rows
            [1]*len(rows)           # coefficients of rows
        )
        num_cols = mp.getNumCol()
        assert num_cols == len(columns_), f"{iter} {solve_exact} - {num_cols} != {len(columns_)}"

        # Solves the master problem
        mp.run()

    # return m, columns_, list(solution.col_value)


#
# Branching helper structures/functions
#
class Node:
    layer: int
    branching_rule: list[tuple[tuple[int, int], float, int]]

    value: float
    final_columns: list[list[int]]
    final_columns_vals: list[float]
    final_fractional_columns: list[int]

    def __init__(self, layer: int, new_branching_rule: tuple[tuple[int, int], float, int] = None, parent = None):
        self.layer = layer
        self.parent = parent
        self.branching_rule = [] if new_branching_rule is None or self.parent is None else self.parent.branching_rule + [new_branching_rule]


def getFractionalColumns(vals: list, columns: list):
    return sorted(
        [k for k, val in enumerate(vals) if abs(round(val) - val) > 1e-6],
        key=lambda k: -len(columns[k]) # order by most fractional (largest number of items)
    )

def getRowsToBranch(node: Node):
    rows = []
    branching_pairs = [el[0] for el in node.branching_rule]
    for r, s in combinations(range(NumberItems), 2):
        if (r,s) not in branching_pairs and (s,r) not in branching_pairs:
            cols_with_both = set(
                idx 
                for idx, column in enumerate(node.final_columns)
                if r in column and s in column
            )
            if len(cols_with_both) > 0:
                sum_columns = sum(node.final_columns_vals[idx] for idx in cols_with_both)
                if 1e-6 < sum_columns < 1-1e-6:
                    rows += [(r, s, cols_with_both, sum_columns)]

    return sorted(rows, key = lambda el: -el[3])


def solveNode(node: Node, columns: list[list[int]], model: highspy.Highs):
    new_column_set = [column.copy() for column in columns]

    count_columns = len(new_column_set)
    num_cols = model.getNumCol()
    assert count_columns == num_cols, f"{num_cols} != {count_columns}"

    # Retrives the feasible columns and turns off all the infeasible ones
    feasible_columns = getFeasibleColumns(columns, node.branching_rule)
    upper_bounds = [0 if idx not in feasible_columns else 1 for idx in range(len(columns))]

    model.changeColsBounds(
        len(new_column_set),                       # Qty columns to change bounds
        list(range(len(new_column_set))),          # Which columns
        [0]*len(new_column_set),                   # Lower bound (0 if not assigned else 1)
        upper_bounds                        # Upper bound (always 1)
    )

    model.run()

    if model.getModelStatus() == highspy.HighsModelStatus.kOptimal:
        count_columns = len(new_column_set)
        num_cols = model.getNumCol()
        assert count_columns == num_cols, f"{num_cols} != {count_columns}"

        # try:
        
        # model, new_column_set, vals = generateColumns(new_column_set, model, msg=False, solve_exact=False, branching_rule=node.branching_rule) # Generate columns quickly

        # count_columns = len(new_column_set)
        # num_cols = model.getNumCol()
        # assert len(new_column_set) == num_cols, f"{num_cols} != {count_columns}"
        
        model, new_column_set, vals = generateColumns(new_column_set, model, msg=False, solve_exact=True, branching_rule=node.branching_rule)  # Prove optimality on node

        count_columns = len(new_column_set)
        num_cols = model.getNumCol()
        assert len(new_column_set) == num_cols, f"{num_cols} != {count_columns}"
        
        # if len(new_column_set) > count_columns:
        node.value = sum(vals)
        node.final_columns = new_column_set
        node.final_columns_vals = vals
        node.final_fractional_columns = getFractionalColumns(vals, new_column_set)
    
        return True, model, node
        # except Exception as e:
        #     print(str(e))
        #     False, model, node

    return False, model, node


#
# Branch-and-price algorithm
#
# Note: Branch selection has been chosen for simple implementation and it's ability to avoid symmetry. 
# This comes at the cost of an unbalanced search tree (other approaches may solve the problem with fewer branches), 
# and it is less likely to be generalizable to other problems.
#
# Specifically, the code "up-branches" columns with fractional value in the RMP solution, i.e., forces specific columns 
# to be selected in RMP.  The subproblems don't need to explicitly enforce this constraint (unlike other branching strategies).
#
def branchAndPrice(m, vals, columns, start_time=0):
    log_start_time = time.perf_counter()
    header = ["NodesExpl", "TreeSize", "CurrCols", "BestFracCols", "UB", "Time"]
    row_format ="".join(["{:>"+f"{4+len(title)}"+"}" for title in header])
    print(row_format.format(*header))

    root_node = Node(-1)
    root_node.value = sum(vals)
    root_node.final_columns = columns
    root_node.final_columns_vals = vals
    root_node.final_fractional_columns = getFractionalColumns(vals, columns)

    # RMP lower bound can be rounded up to nearest integer (avoiding floating point precision issues)
    # so, take integer floor, add one if fractional part greater than tolerance
    rmp_LB = int(root_node.value) + int(root_node.value % 1 > 1e-6)
    best_obj = math.inf
    best_node = None
    best_count_frac_columns = math.inf
    count_nodes_visited = 0

    # if no fractional columns, then root node is an optimal integer solution
    if len(root_node.final_fractional_columns) == 0:
        best_node = root_node
        print(row_format.format(*[0, 0, len(columns), 0, best_node.value, f"{round(time.perf_counter()-start_time, 2) :.2f}" ]))

    # create initial branches for each column with fractional value in RMP solution
    branch_tree: list[Node] = []
    for r, s, _, integrality in getRowsToBranch(root_node):
        for rule in range(2):
            branch_tree += [Node(0, ((r, s), integrality, rule), root_node)]

    # Adds cut 
    m.addRow(
        rmp_LB,                                              # lhs
        highspy.kHighsInf,                                   # rhs
        len(columns),                                        # Number of non-zero variables
        list(range(len(columns))),                           # Indexes of variable
        [1] * len(columns)                                   # Coefficients of variables
    )
    # branch_tree_dict = {frozenset(node.assigned_columns): node for node in branch_tree}

    # explore branches, looking for optimal integer solution
    while len(branch_tree) > 0:
        # Choose next node in branch to evaluate.  Prioritize nodes that are likely to be 
        # integer (DFS with fewer fractional columns), and likely to be optimal (larger value).
        #
        # i.e., maximize branch depth (layer), minimize number of fractional columns of its parent, 
        # and maximize value of its original fractional columns.
        # last_layer = max(node.layer for node in branch_tree)
        node = min(branch_tree, key=lambda node: (
            -node.layer,
            -node.branching_rule[-1][2] if len(node.branching_rule) > 0 else 0,
            # len(node.parent.final_fractional_columns)/len(node.parent.final_columns) if node.parent is not None else math.inf,
            abs(round(node.branching_rule[-1][1]) - node.branching_rule[-1][1]) if len(node.branching_rule) > 0 else math.inf,
            # -node.branching_rule[-1][2] if len(node.branching_rule) > 0 else 0,
        ))
        # print(node.branching_rule)
        # input()
        branch_tree.remove(node)

        # Solves the node with column generation
        mp_is_feasible, m, node = solveNode(node, columns, m)
        if mp_is_feasible and len(node.final_columns) > len(columns):
            columns = node.final_columns

        count_nodes_visited += 1

        # Only add columns if the master problem is feasible
        # Prune nodes that cannot lead to optimal solution        
        if mp_is_feasible and node.value < best_obj:
            count_vals = len(m.getSolution().col_value)
            count_columns = len(columns)
            assert count_vals == count_columns, f"{count_vals} != {count_columns}"

            count_fractional_columns = len(node.final_fractional_columns)

            # if no fractional columns, then this is an integer solution
            # update the best solution here for console output
            if count_fractional_columns == 0 and node.value < best_obj:
                best_obj = int(round(node.value, 0))  # avoid float precision issues
                best_node = node
                
                # stop if we've found a provably optimal solution (using lower bound from RMP root node)
                if rmp_LB == best_obj:
                    break

            # found a less fractional solution or enough time has elapsed
            if count_fractional_columns < best_count_frac_columns or time.perf_counter()-log_start_time > 5:
                best_count_frac_columns = min(best_count_frac_columns, count_fractional_columns)
                print(row_format.format(*[
                    count_nodes_visited,
                    len(branch_tree),
                    len(columns),
                    best_count_frac_columns,
                    best_obj if best_obj < math.inf else "-",
                    f"{round(time.perf_counter()-start_time, 2) :.2f}"
                ]))
                log_start_time = time.perf_counter()

            # add all fractional columns as new branches
            if count_fractional_columns > 0:
                for r, s, _, integrality in getRowsToBranch(node):
                    for rule in range(2):
                        new_node = Node(node.layer + 1, ((r, s), integrality, rule), node)
                        branch_tree += [new_node]
                

                # for column_idx in node.final_fractional_columns:
                #     new_assigned_columns = node.assigned_columns + [column_idx]
                #     key = frozenset(new_assigned_columns)
                    
                #     if key not in branch_tree_dict:
                #         new_node = Node(node.layer + 1, new_assigned_columns, node)
                #         branch_tree = [new_node] + branch_tree
                #         branch_tree_dict[key] = new_node

            

    # explored the entire tree, so best found solution is optimal
    if best_node is not None:
        return [column for idx, column in enumerate(best_node.final_columns) if best_node.final_columns_vals[idx] > 0.9]
    
    raise Exception("It should not get to this point")


if __name__ == '__main__':
    if max(ItemWeights) > BinCapacity:
        print(f"Instance is infeasible: item {max(enumerate(ItemWeights), key=itemgetter(1))[0]} has weight {max(ItemWeights)} > {BinCapacity} (bin capacity).")
        exit()

    start_time = time.perf_counter()
    greedy_bins = solveGreedyModel()
    greedy_time = time.perf_counter()-start_time
    print(f"Greedy estimate: {len(greedy_bins)} bins")
    print(f"Finished in {greedy_time: .2f}s\n")

    # start_time = time.perf_counter()
    # compact_bins = solveCompactModel()
    # compact_time = time.perf_counter()-start_time

    # print(f"\nSolution by compact model: {len(compact_bins)} bins")
    # for bin, items in enumerate(compact_bins):
    #     tt_weight = round(sum(ItemWeights[i] for i in items))
    #     print(f"Bin {bin+1} ({tt_weight} <= {BinCapacity}): {items}")
    #     assert tt_weight <= BinCapacity

    # print(f"Finished in {compact_time: .2f}s\n")

    # start with initial set of columns for feasible master problem
    start_time = time.perf_counter()
    columns = [[i] for i in range(NumberItems)] + greedy_bins
    m, vals, duals = createMasterProblem(columns)
    
    print("\nSolving root node:")
    m, columns, vals = generateColumns(columns, m, solve_exact=False, start_time=start_time)
    
    print("\nProving optimality on root node:")
    m, columns, vals = generateColumns(columns, m, solve_exact=True, start_time=start_time)

    print("\nBranch-and-price:")
    CG_GAP_REL = 1e-3
    cg_bins = branchAndPrice(m, vals, columns, start_time)
    cg_time = time.perf_counter()-start_time

    print(f"\nSolution by column generation: {len(cg_bins)} bins")
    for bin, items in enumerate(cg_bins):
        tt_weight = round(sum(ItemWeights[i] for i in items))
        print(f"Bin {bin+1} ({tt_weight} <= {BinCapacity}): {items}")
        assert tt_weight <= BinCapacity

    print(f"Finished in {cg_time: .2f}s\n")

    print(f"Greedy : {len(greedy_bins)} bins, {greedy_time:6.2f}s")
    # print(f"Compact: {len(compact_bins)} bins, {compact_time:6.2f}s")
    print(f"ColGen : {len(cg_bins)} bins, {cg_time:6.2f}s\n")

