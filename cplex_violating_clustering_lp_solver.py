import numpy as np
from cplex import Cplex
from scipy.spatial.distance import pdist,squareform
import time

'''
The main function in this file is violating_clustering_lp_solver.
This function takes as input a collection of data points, the number of desired
clusters, a list of colors of each points, and fairness parameters.
It then constructs the fair clustering lp and solves it. It returns 
a fractional solution to the fair clustering lp.

Input Description:
    df: a dataframe of the input points
    num_centers: number of cluster that we want to create
    color_flag : a list of color values for all the points -- helpful in adding constraints to the lp solver
    alpha: dict where the keys are colors and the values is the alpha for that color
    beta: dict where the keys are colors and the values are the beta for that color

Output Description:
    res: a dictionary with the following keys:
        "status": an integer code depicting the outcome of the lp solver
        "success": a string to interpret the above code
        "objective": the objective function value
        "assignment": the assignment of values to all the LP variables

Reading the "assignment" array:
    the variables are created in the following order:
        for all j in points
            for all i in centers 
                create x_{j}_{i} 
        for all i in points
            create y_i
    So the "assignment" array, which is a list of floats, corresponds to this variable order
'''


def violating_lp_clustering(df, num_centers, alpha, beta, color_flag, clustering_method, violation):

    if clustering_method == "kmeans" or clustering_method == "kmedian":
        cost_fun_string = 'euclidean' if clustering_method == "kmedian" else 'sqeuclidean'
        problem, objective = violating_clustering_lp_solver(df, num_centers, color_flag, alpha, beta, violation, cost_fun_string)
        # Step 5. call the solver

        t1 = time.monotonic()
        problem.solve()
        t2 = time.monotonic()
        print("LP solving time = {}".format(t2-t1))

        objective_value = problem.solution.get_objective_value()
        if clustering_method == "kmeans":
            objective_value = np.sqrt(objective_value)
        # problem.solution is a weakly referenced object, so we must save its data
        #   in a dictionary so we can write it to a file later.
        res = {
            "status": problem.solution.get_status(),
            "success": problem.solution.get_status_string(),
            "objective": objective_value,
            "assignment": problem.solution.get_values(),
        }

        return res

    elif clustering_method == "kcenter":
        problem, objective = violating_clustering_lp_solver(df, num_centers, color_flag, alpha, beta, violation, 'sqeuclidean')
        cost_ub = max(objective) + 1
        cost_lb = 0
        lowest_feasible_cost = cost_ub
        cheapest_feasible_lp = problem

        while cost_ub > cost_lb + 0.1:
            cost_mid = (cost_ub + cost_lb)/2.0
            new_problem, new_objective = violating_clustering_lp_solver(df, num_centers, color_flag, alpha, beta, violation, 'sqeuclidean')
            to_delete = [idx for idx, el in enumerate(objective) if el > cost_mid]
            # print("to delete : {}".format(to_delete))
            if len(to_delete) > 0:
                new_problem.variables.delete(to_delete)

            new_problem.solve()
            new_stats = new_problem.solution.get_status()

            if new_stats == 1: # optimal!
                cost_ub = cost_mid
                lowest_feasible_cost = cost_mid
                cheapest_feasible_lp = new_problem
            else:
                cost_lb = cost_mid

            # else:
            #     raise ValueError("LP solver stat code {}".format(new_stats) + " with cost {}".format(cost_mid))

        # to_delete = [idx for idx, el in enumerate(objective) if el > lowest_feasible_cost]
        # if len(to_delete) > 0:
        #     problem.variables.delete(to_delete)
        # problem.solve()

        # problem.solution is a weakly referenced object, so we must save its data
        #   in a dictionary so we can write it to a file later.
        res = {
            "status" : cheapest_feasible_lp.solution.get_status(),
            "success" : cheapest_feasible_lp.solution.get_status_string(),
            "objective" : np.sqrt(lowest_feasible_cost),
            "assignment" : cheapest_feasible_lp.solution.get_values(),
        }

        return res

    else:
        print("Not a valid clustering method. Available methods are: " 
              "\'kmeans\', \'kmedian\', and \'kcenter\'.")
        return None



def violating_clustering_lp_solver(df, num_centers, color_flag, alpha, beta, violation, cost_fun_string):


    # There are primarily five steps:
    # 1. Initiate a model for cplex
    # 2. Declare if it is minimization or maximization problem
    # 3. Add variables to the model. The variables are generally named.
    #    The upper bounds and lower bounds on the range for the variables
    #    are also mentioned at this stage. The coefficient of the objective
    #    functions are also entered at this step
    # 4. Add the constraints to the model. The constraint matrix, denoted by A,
    #    can be added in three ways - row wise, column wise or non-zero entry wise.
    # 5. Finally, call the solver.

    # Step 1. Initiate a model for cplex.

    problem = Cplex()

    # Step 2. Declare that this is a minimization problem

    problem.objective.set_sense(problem.objective.sense.minimize)

    # Step 3.   Declare and  add variables to the model. The function
    #           prepare_to_add_variables prepares all the
    #           required information for this stage.
    #
    #    objective: a list of coefficients (float) in the linear objective function
    #    lower bounds: a list of floats containing the lower bounds for each variable
    #    upper bounds: a list of floats containing the upper bounds for each variable
    #    variable_name: a list of strings that contains the name of the variables

    print("Starting to add variables...")
    t1 = time.monotonic()
    objective, lower_bounds, upper_bounds, variable_names = prepare_to_add_variables(df, cost_fun_string)
    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          names=variable_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding variable = {}".format(t2-t1))

    # Step 4.   Declare and add constraints to the model.
    #           There are few ways of adding constraints: rwo wise, col wise and non-zero entry wise.
    #           The function prepare_to_add_constraints_by_entry
    #           prepares the required data for this step. Assume the constraint matrix is A.
    #  constraints_row: Encoding of each row of the constraint matrix
    #  senses: a list of strings that identifies whether the corresponding constraint is
    #          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
    #  rhs: a list of floats corresponding to the rhs of the constraints.
    #  constraint_names: a list of string corresponding to the name of the constraint

    print("Starting to add constraints...")
    t1 = time.monotonic()
    objects_returned = prepare_to_add_constraints(df, num_centers, color_flag, beta, alpha, violation)
    print(objects_returned)
    constraints_row, senses, rhs, constraint_names = objects_returned

    problem.linear_constraints.add(lin_expr=constraints_row,
                                   senses=senses,
                                   rhs=rhs,
                                   names=constraint_names)
    t2 = time.monotonic()
    print("Completed. Time for creating and adding constraints = {}".format(t2-t1))

    # Optional: We can set various parameters to optimize the performance of the lp solver
    # As an example, the following sets barrier method as the lp solving method

    problem.parameters.lpmethod.set(problem.parameters.lpmethod.values.barrier)
    #problem.parameters.mip.limits.nodes.set(5000)

    return problem, objective


# This function creates necessary stuff to prepare for variable
# addition to the cplex lp solver
# Input: See the input description of the main function
# Output:
#   objective: a list of coefficients (float) in the linear objective function
#   lower bounds: a list of floats containing the lower bounds for each variable
#   upper bounds: a list of floats containing the upper bounds for each variable
#   variable_name: a list of strings that contains the name of the variables
# The ordering of variables are as follows: for every point, we create #centers many
# variables and put them together before moving onto the next point.
# for j in range(num_points)
#   for i in range(num_points)
#       create x_j_i
# for i in range(num_points)
#       create y_i


def prepare_to_add_variables(df, cost_fun_string):

    num_points = len(df)

    # two types of variable: assignment (x_ij) variable and facility opening (y_i) variables
    variable_assn_names = ["x_{}_{}".format(i, j) for i in range(num_points) for j in range(num_points)]
    variable_facility_names = ["y_{}".format(i) for i in range(num_points)]
    variable_names = variable_assn_names + variable_facility_names

    total_variables = num_points * num_points + num_points

    # All values should be between 0 and 1 -- if a single tuple is provided,
    #   then it will be applied to all points, according to scipy docs.
    lower_bounds = [0 for _ in range(total_variables)]
    upper_bounds = [1 for _ in range(total_variables)]

    # Cost function: Minimize the weighted sum of the distance from each point
    #   to each center.
    objective = cost_function(df, cost_fun_string)

    return objective, lower_bounds, upper_bounds, variable_names

# Cost function: Minimize the weighted sum of the distance from each point
#   to each center.
# Implementation details:
#   pdist(X, metric='euclidean', *args, **kwargs)[source]
#          Pairwise distances between observations in n-dimensional space.
#       Y : ndarray
#       Returns a condensed distance matrix Y.
#       For each i and j (where i < j < m),where m is the number of original observations.
#       The metric dist(u=X[i], v=X[j]) is computed and stored in entry ij.
#   squareform converts between condensed distance matrices and square distance matrices.
#   ravel(a,order='C'): A 1-D array, containing the elements of the input, is returned. A copy is made only if needed.
#   Note that all_pair_distance is an array and we need to convert it to a list before returning


def cost_function(df, cost_fun_string):
    all_pair_distance = pdist(df.values,cost_fun_string)
    all_pair_distance = squareform(all_pair_distance)
    all_pair_distance = all_pair_distance.ravel().tolist()
    pad_for_facility = [0]* len(df)
    return all_pair_distance + pad_for_facility

# This function prepares for constraints addition by non zero entries
# Input: See the input description of fair_partial_assignment_lp_by_nonzero()
# Output:
#  constraints_row: Encoding of each row of the constraint matrix
#  senses: a list of strings that identifies whether the corresponding constraint is
#          an equality or inequality. "E" : equals to (=), "L" : less than (<=), "G" : greater than equals (>=)
#  rhs: a list of floats corresponding to the rhs of the constraints.
#  constraint_names: a list of string corresponding to the name of the constraint
# There are four types of constraints.
#   1. Assignment constraints: every client is assigned to exactly one facility
#           we have no_of_points*no_of_points many assignment constraints (sum_{i} x_j_i = 1 for all j)
#   2. Assignment validity constraints: x_j_i <= y_i for all i and j
#           we have no_of_points*no_of_points many validity constraints
#   3. Facility constraints -- at most k facilities are open (\sum_{i} y_i \leq k)
#   4. Fairness constraints: finally we have then we have num_points*num_colors*2 many fairness constraints.
#          the fairness constraints are indexed by center, and then color. We first have all beta constraints, followed
#          by alpha constraints


def prepare_to_add_constraints(df, num_centers, color_flag, beta, alpha, violation):

    num_points = len(df)

    # The following steps constructs various types of constraints.
    sum_constraints, sum_rhs = constraint_sums_to_one(num_points)

    validity_constraints, validity_rhs = constraint_validity(num_points)

    facility_constraints, facility_rhs = constraint_facility(num_points, num_centers)

    # We now combine all these types of constraints
    constraints_row = sum_constraints + validity_constraints + facility_constraints
    rhs = sum_rhs + validity_rhs + facility_rhs

    num_equality_constraints = len(sum_rhs)


    for var in color_flag:
        var_color_flag, var_beta, var_alpha = color_flag[var], beta[var], alpha[var]
        color_constraint, color_rhs = constraint_color(num_points, var_color_flag, var_beta, var_alpha, violation)
        constraints_row.extend(color_constraint)
        rhs.extend(color_rhs)

    num_inequality_constraints = len(rhs) - num_equality_constraints

    # The assignment constraints are of equality type and the rest are less than equal to type
    senses = ["E" for _ in range(num_equality_constraints)] + ["L" for _ in range(num_inequality_constraints)]

    # Name the constraints
    constraint_names = ["c_{}".format(i) for i in range(len(rhs))]

    return constraints_row, senses, rhs, constraint_names

# this function adds the constraint that every client must be assigned to exactly one center
# Implementation:
# Example: assume 4 points. Then we have 16 variables x_i_j types.
# The assignment constraints are entered in the following format:
# [[['x_0_0', 'x_0_1', 'x_0_2', 'x_0_3'], [1, 1, 1, 1]],
#  [['x_1_0', 'x_1_1', 'x_1_2', 'x_1_3'], [1, 1, 1, 1]],
#  [['x_2_0', 'x_2_1', 'x_2_2', 'x_2_3'], [1, 1, 1, 1]],
#  [['x_3_0', 'x_3_1', 'x_3_2', 'x_3_3'], [1, 1, 1, 1]]]


def constraint_sums_to_one(num_points):

    constraints = [[["x_{}_{}".format(j,i) for i in range(num_points)], [1] * num_points] for j in range(num_points)]
    rhs = [1] * num_points
    return constraints, rhs

# This function adds the constraints of type: x_ji - y_i <= 0 for all i and j
# [[['x_0_0', 'y_0'], [1, -1]],
#  [['x_0_1', 'y_1'], [1, -1]],
#  [['x_0_2', 'y_2'], [1, -1]],
#  [['x_0_3', 'y_3'], [1, -1]],
#  [['x_1_0', 'y_0'], [1, -1]],
#  [['x_1_1', 'y_1'], [1, -1]],
#  [['x_1_2', 'y_2'], [1, -1]],
#  [['x_1_3', 'y_3'], [1, -1]]]


def constraint_validity(num_points):

    constraints = [[["x_{}_{}".format(j,i),"y_{}".format(i)], [1,-1]] for j in range(num_points) for i in range(num_points)]
    rhs = [0]* (num_points*num_points)
    return constraints, rhs

# This function add the constraint sum_{i} y_i <= k
# Assume there are 4 points.
# [['y_1','y_2','y_3','y_4'][1,1,1,1]]


def constraint_facility(num_points, num_centers):
    constraints = [[["y_{}".format(i) for i in range(num_points)], [1]*num_points]]
    rhs = [num_centers]
    return constraints, rhs

# this function adds the fairness constraint
# the details are similar to the constraint_color function in the
# cplex_Fair_Assignment_lp_solver file.
#TODO may be add an example here later

def constraint_color(num_points, color_flag, beta, alpha, violation):

    beta_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [beta[color] - 1 if color_flag[j] == color else beta[color] for j in range(num_points)]]
                        for i in range(num_points) for color, _ in beta.items()]
    alpha_constraints = [[["x_{}_{}".format(j, i) for j in range(num_points)],
                         [np.round(1 - alpha[color], decimals=3) if color_flag[j] == color else (-1) * alpha[color]
                          for j in range(num_points)]]
                            for i in range(num_points) for color, _ in beta.items()]
    constraints = beta_constraints + alpha_constraints
    number_of_constraints = num_points * len(beta) * 2
    rhs = [violation]*number_of_constraints
    return constraints, rhs
