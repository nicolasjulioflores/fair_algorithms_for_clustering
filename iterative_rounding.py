import numpy as np
from collections import defaultdict
import math
from scipy.spatial.distance import cdist
from cplex import Cplex
import time


def iterative_rounding_lp(df, centers, distance, color_flag, res):

    num_points = len(df)
    num_centers = len(centers)
    lp_assignment = res["assignment"]

    # Process the lp_assignment array, and store fractional assignments separately.
    # Since the distance array already contains all possible distances, we only
    # need to store the index of the points that are still fractionally assigned
    # to multiple facilities.
    # integral_assignment:
    # frac_list: A list of indices of the points that are not integrally assigned
    # frac_lp_assgn: A list of lists of size frac_list x num_centers
    #                For every point in frac_list, it stores the corresponding LP assignments.
    #                Some of the entries might be 0. The order of the points is same
    #                in frac_list and frac_lp_assgn

    objects_returned = preprocess_lp_solution(lp_assignment, num_points, num_centers)
    integral_assignment, frac_list, frac_lp_assgn = objects_returned

    # The following part is for debugging purposes
    obj_ret = find_lp_cost(lp_assignment,distance,num_centers,num_points,frac_list)
    initial_cost, ini_frac_list_cost, ini_frac_list_cost_item = obj_ret
    print("Initial LP cost ={}, also {}, frac cost = {}".format(res["objective"],initial_cost,ini_frac_list_cost))
    print_list = [[item,frac_lp_assgn[ind],ini_frac_list_cost_item[ind]] for ind,item in enumerate(frac_list)]
    for item in print_list:
        print(item)
    ini_frac_list = [j for j in frac_list]

    # Assume frac_list is not empty. Here the code is optimized
    # with the assumption that frac_list is always very small compared to num_points
    # We need to create an instance of the mbdmb problem, and solve the corresponding lp.
    # We keep repeating the steps till all the assignments become integral
    # res:
    # new_lp_assignment:
    # integral_assignment:
    # frac_list:
    # frac_lp_assgn:

    while len(frac_list) != 0:
        res = mbdmb_lp_solver(distance, num_centers, frac_list, frac_lp_assgn, color_flag)
        new_lp_assignment = res["assignment"]
        objects_returned = update_assignment(num_centers, frac_list, frac_lp_assgn, new_lp_assignment, integral_assignment)
        integral_assignment, frac_list, frac_lp_assgn = objects_returned

    final_cost =  find_integral_cost(integral_assignment, distance, num_centers)
    res["assignment"] = reformat_assignment (integral_assignment, num_centers)
    res["objective"] = final_cost
    print("Final LP cost ={}, and {}".format(res["objective"],final_cost))

    print_list = [[item, integral_assignment[item], ini_frac_list_cost_item[ind]] for ind, item in enumerate(ini_frac_list)]
    for item in print_list:
        print(item)

    return res


def preprocess_lp_solution(lp_assignment, num_points, num_centers):

    assignment = np.array(lp_assignment)
    assignment = np.reshape(assignment, (num_points, num_centers))

    # For every point that is integrally assigned to some center (x_j_i=1 for point j and center i)
    # the assignment is final. We capture this in the assignment function. The points
    # that are not assigned to any center yet has -1 in the corresponding entry.

    integral_assignment = np.full(num_points,-1)
    frac_list = []
    frac_lp_assgn = []

    for i in range(num_points):
        if np.max(assignment[i] == 1.0):
            integral_assignment[i] = np.argmax(assignment[i])
        else:
            frac_list.append(i)
            frac_lp_assgn.append(assignment[i].tolist())

    return integral_assignment, frac_list, frac_lp_assgn


def mbdmb_lp_solver(distance, num_centers, frac_list, frac_lp_assgn, color_flag):

    # Step 1. Initiate a model for cplex.

    print("Initializing Cplex model")
    problem = Cplex()

    # Step 2. Declare that this is a minimization problem

    problem.objective.set_sense(problem.objective.sense.minimize)

    # Step 3.   Declare and  add variables to the model.

    print("Starting to add variables...")
    t1 = time.monotonic()
    objects_returned = prepare_to_add_variables(distance, num_centers, frac_list, frac_lp_assgn)
    objective, lower_bounds, upper_bounds, variable_names = objects_returned
    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=variable_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding variable = {}".format(t2-t1))

    # Step 4.   Declare and add constraints to the model.

    print("Starting to add constraints...")
    t1 = time.monotonic()
    objects_returned = prepare_to_add_constraints(num_centers, frac_list, frac_lp_assgn, color_flag)
    constraints_row, senses, rhs, constraint_names = objects_returned
    problem.linear_constraints.add(lin_expr=constraints_row,
                                   senses=senses,
                                   rhs=rhs,
                                   names=constraint_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding constraints = {}".format(t2-t1))

    # Optional: We can set various parameters to optimize the performance of the lp solver
    # As an example, the following sets barrier method as the lp solving method
    # The other available methods are: auto, primal, dual, sifting, concurrent

    #problem.parameters.lpmethod.set(problem.parameters.lpmethod.values.barrier)

    # Step 5. call the solver

    t1 = time.monotonic()
    problem.solve()
    t2 = time.monotonic()
    print("LP solving time time = {}".format(t2 - t1))

    res = {
        "status": problem.solution.get_status(),
        "success": problem.solution.get_status_string(),
        "objective": problem.solution.get_objective_value(),
        "assignment": problem.solution.get_values(),
    }

    return res


def prepare_to_add_variables(distance, num_centers, frac_list, frac_lp_assgn):

    num_points = len(frac_list)
    variable_names = []
    objective = []
    total_variables = 0
    # Name the variables -- x_j_i is set to 1 if j th pt is assigned to ith center
    for index,j in enumerate(frac_list):
        for i in range(num_centers):
            if frac_lp_assgn[index][i] != 0.0:
                variable_names.append("x_{}_{}".format(j,i))
                objective.append(distance[j*num_centers + i])
                total_variables += 1

    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]

    return objective, lower_bounds, upper_bounds, variable_names


def prepare_to_add_constraints(num_centers, frac_list, frac_lp_assgn, color_flag):

    # The following steps constructs the assignment constraint. Each corresponding row in A
    # has #centers many non-zero entries.
    constraints_row, rhs = constraint_sums_to_one(num_centers, frac_list, frac_lp_assgn)
    sum_const_len = len(rhs)

    # The following steps constructs the fairness constraints. There are #centers * # colors * 2
    # many such constraints. Each of them has #points many non-zero entries.

    obj_ret = constraint_fairness(num_centers, frac_list, frac_lp_assgn, color_flag)
    color_constraint, color_rhs = obj_ret
    constraints_row.extend(color_constraint)
    rhs.extend(color_rhs)

    # The assignment constraints are of equality type and the rest are less than equal to type
    senses = ["E" for _ in range(sum_const_len)] + ["L" for _ in range(len(rhs) - sum_const_len)]

    # Name the constraints
    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]

    return constraints_row, senses, rhs, constraint_names

    # this function adds the constraint that every client must be assigned to exactly one center
    # Implementation:
    # Example: assume 3 points and 2 centers. Total of 6 variables: x_0_0,x_0_1 for first point, and so on.
    #  row 0         x00 + x01                 = 1
    #  row 1                  x10 + x11        = 1
    #  row 2                         x20 + x21 = 1
    # The constraints are entered in the following format:
    # [[['x_0_0', 'x_0_1'], [1, 1]],
    #  [['x_1_0', 'x_1_1'], [1, 1]],
    #  [['x_2_0', 'x_2_1'], [1, 1]]]


def constraint_sums_to_one(num_centers, frac_list, frac_lp_assgn):

    constraints, rhs = [], []
    for index,j in enumerate(frac_list):
        variable, coef = [], []
        for i in range(num_centers):
            if frac_lp_assgn[index][i] != 0.0:
                variable.append("x_{}_{}".format(j,i))
                coef.append(1)
        if len(variable) != 0: # why should it be? no reason, but just in case!
            constraints.append([variable,coef])
            rhs.append(1)

    # constraints = [
    #               [ ["x_{}_{}".format(j, i) for i in range(num_centers)], [1] * num_centers ]
    #               for j in frac_list
    #               if frac_lp_assgn[index][i] != 0.0
    #               ]
    # rhs = [1] * len(frac_list)
    return constraints, rhs

    # this function adds the fairness constraint
    # Implementation:
    # The following example demonstrates this.
    # These constraints are in effect only if number of elements is more than 2(DELTA + 1)
    # Example:  Assume 3 points and 2 centers and 2 colors.
    #           Total of 6 variables: x00,x01 for first point, x10,x11 for second point and so on.
    #           Assume 1 and 2 belong to color 0. Let a == alpha and b == beta
    #
    #   The first type of constraint is a center constraint
    #       x00             +  x10            +  x20                 <= ceil(T_f)      center 0
    #     - x00             -  x10            -  x20                 <= -floor(T_f)    center 0
    #       x01             +  x11            +  x21                 <= ceil(T_f)      center 1
    #     - x01             -  x11            -  x21                 <= -floor(T_f)    center 1
    #
    #   The second type is for every center and every color
    #                          x10            +  x20                 <= ceil(T_f_0)    center 0, color 0
    #       x00                                                      <= ceil(T_f_1)    center 0, color 1
    #                       -  x10            -  x20                 <= -floor(T_f_0)  center 0, color 0
    #    -  x00                                                      <= -floor(T_f_1)  center 0, color 1

    # Below we depict the details of the entries (the first 2 rows)
    # [
    # [['x_0_0','x_1_0','x_2_0'],[1,1,1]]
    # [['x_0_0','x_1_0','x_2_0'],[-1,-1,-1]]
    # ...]


def constraint_fairness(num_centers, frac_list, frac_lp_assgn, color_flag):

    num_points = len(frac_list)
    DELTA = len(color_flag)+1

    # for every facility i, T_f[i] stores the total fractional assignment
    assignment = np.array(frac_lp_assgn)
    T_f = assignment.sum(axis=0)
    T_f_ceil = np.ceil(T_f)
    T_f_floor = np.floor(T_f)

    # First add the center constraints
    # A constraint is added if strictly more than 2*DELTA many variables are there
    distinct_f = np.count_nonzero(assignment,axis=0)
    center_constraints, center_rhs = [], []

    for i in range(num_centers):
        if distinct_f[i] > 2 * DELTA:
            variable = []
            for index, j in enumerate(frac_list):
                if frac_lp_assgn[index][i] != 0.0:
                    variable.append("x_{}_{}".format(j, i))

            coef = [1] * len(variable)
            center_constraints.append([variable, coef])
            center_rhs.append(T_f_ceil[i])

            coef = [-1] * len(variable)
            center_constraints.append([variable, coef])
            center_rhs.append(T_f_floor[i])

    T_f_l = {}
    distinct_f_l = {}
    constraints_variable_f_l = {}
    for var in color_flag:
        T_f_l[var] = defaultdict(lambda: defaultdict(float))
        distinct_f_l[var] = defaultdict(lambda: defaultdict(int))
        constraints_variable_f_l[var] = defaultdict(lambda: defaultdict(list))
        var_color_flag = color_flag[var]
        for index, j in enumerate(frac_list):
            for i in range(num_centers):
                # add x_point_f to T_f_l for each color class l to which point belongs
                if frac_lp_assgn[index][i] != 0.0:
                    color = var_color_flag[j]
                    T_f_l[var][color][i] += assignment[index][i]
                    distinct_f_l[var][color][i] += 1
                    constraints_variable_f_l[var][color][i].append("x_{}_{}".format(j, i))

    # now add the color and center combined fairness constraints
    color_center_constraints, color_center_rhs = [], []
    for var,color_dict in T_f_l.items():
        for color in color_dict:
            for i in range(num_centers):
                if distinct_f_l[var][color][i] > 2 * DELTA:

                    coef_ub = [1] * len(constraints_variable_f_l[var][color][i])
                    color_center_constraints.append([constraints_variable_f_l[var][color][i], coef_ub])
                    color_center_rhs.append(math.ceil(T_f_l[var][color][i]))

                    coef_lb = [-1] * len(constraints_variable_f_l[var][color][i])
                    color_center_constraints.append([constraints_variable_f_l[var][color][i], coef_lb])
                    color_center_rhs.append(-math.floor(T_f_l[var][color][i]))

    constraints = center_constraints + color_center_constraints
    rhs = center_rhs + color_center_rhs
    return constraints, rhs


def update_assignment(num_centers, frac_list, frac_lp_assgn, new_lp_assignment, integral_assignment):

    squared_lp_assignment = []
    k = 0
    for index,j in enumerate(frac_list):
        assgn = [0] * num_centers
        for i in range(num_centers):
            if frac_lp_assgn[index][i] != 0.0:
                assgn[i] = new_lp_assignment[k]
                k += 1
            else:
                assgn[i] = 0.0
        squared_lp_assignment.append(assgn)

    num_points = len(frac_list)
    assignment = np.array(squared_lp_assignment)
    assignment = np.reshape(assignment, (num_points, num_centers))

    new_frac_list = []
    new_frac_lp_assgn = []

    for index,j in enumerate(frac_list):
        if np.max(assignment[index] == 1.0):
            integral_assignment[j] = np.argmax(assignment[index])
        else:
            new_frac_list.append(j)
            new_frac_lp_assgn.append(assignment[index].tolist())

    return integral_assignment, new_frac_list, new_frac_lp_assgn


def find_integral_cost (integral_assignment, distance, num_centers):

    cost = 0
    for i in range(len(integral_assignment)):
        cost += distance[i*num_centers + integral_assignment[i]]

    return cost


def reformat_assignment (integral_assignment, num_centers):
    assignment = []
    for i in range(len(integral_assignment)):
        assgn = [0.0]* num_centers
        assgn [integral_assignment[i]] = 1.0
        assignment.extend(assgn)
    return assignment


def find_lp_cost (lp_assignment, distance, num_centers, num_points, frac_list):

    cost, frac_list_cost = 0 , 0
    assignment = np.array(lp_assignment)
    assignment = np.reshape(assignment, (num_points, num_centers))
    for j in range(num_points):
        for i in range(num_centers):
            cost += distance[j*num_centers + i]* assignment[j][i]

    frac_list_cost_item = []
    for j in frac_list:
        cost_item = []
        for i in range(num_centers):
            frac_list_cost += distance[j*num_centers + i]* assignment[j][i]
            cost_item.append(distance[j*num_centers + i])
        frac_list_cost_item.append(cost_item)

    return cost, frac_list_cost,frac_list_cost_item
