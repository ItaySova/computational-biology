# Authors: Dafna Shlomi, Itay Sova

import sys
from random import randint, random, seed

# the amount of matrices in each generation
MAT_AMOUNT = 100
# the number of generations
GENS_NUM = 1500
# algo type
# ALGO = "Normal"
# ALGO = "Darwin"
ALGO = "Lamarck"
# max probability for mutations
MAX_PROB = 0.3
# num of crossovers:
CROSS_NUM = MAT_AMOUNT - 10
# num of replications of the best two metrics in gen
REP_NUM = 5
# used in finding best fitness
MAXIMUM = 10000
# dictionary that will keep the initial numbers and their locations (global)
initial_digits_dict = {}
# dictionary that will keep the locations of gt signs (keys represent the greater value cell and the
# value is the smaller cell) (global)
gt_signs_coords = {}
# later the gt_signs_coords will be split into 4 lists - gt and lt (horizontal and lateral)
lt_signs_coords_horizontal = {}
lt_signs_coords_lateral = {}
gt_signs_coords_horizontal = {}
gt_signs_coords_lateral = {}
# dictionaries for find_wrong_bigger_than2 func:
gt_coords_cols = {}
gt_coords_rows = {}


# this func checks the number of unique digits in a row
def rows_checker(matrix, size):
    rows_score = []
    digits_to_be_in_row = [x for x in range(1, size + 1)]
    digits_spotted = {}
    for i in digits_to_be_in_row:
        digits_spotted[i] = 0
    for i in range(size):
        row_sum = 0
        for j in range(size):
            current_digit = matrix[i][j]
            if digits_spotted[current_digit] == 0:
                row_sum = row_sum + 1
                digits_spotted[current_digit] = 1
        rows_score.append(row_sum)
        for k in digits_to_be_in_row:
            digits_spotted[k] = 0
    return rows_score


# this func checks the number of unique digits in a column
def column_checker(matrix, size):
    column_score = []
    digits_to_be_in_column = [x for x in range(1, size + 1)]
    digits_spotted = {}
    for i in digits_to_be_in_column:
        digits_spotted[i] = 0

    for i in range(size):
        column_sum = 0
        for j in range(size):
            current_digit = matrix[j][i]
            if digits_spotted[current_digit] == 0:
                column_sum = column_sum + 1
                digits_spotted[current_digit] = 1
        column_score.append(column_sum)
        for k in digits_to_be_in_column:
            digits_spotted[k] = 0
    return column_score


# this func returns a list of the wrong < / > in the board
def find_wrong_bigger_than2(mat, type, opt_lst, size=None):
    if type == "rows":
        gt_sings_dict = gt_signs_coords_horizontal
        lt_sings_dict = lt_signs_coords_horizontal
        lt_sings_dict_for_comp = dict((value, key) for key, value in lt_signs_coords_horizontal.items())
        signs_list = list(gt_signs_coords_horizontal.keys())
        signs_list.extend(lt_signs_coords_horizontal.values())
    else:
        gt_sings_dict = gt_signs_coords_lateral
        lt_sings_dict = lt_signs_coords_lateral
        lt_sings_dict_for_comp = dict((value, key) for key, value in lt_signs_coords_lateral.items())
        signs_list = list(gt_signs_coords_lateral.keys())
        signs_list.extend(lt_signs_coords_lateral.values())
    wrong_bigger_counter = 0
    loop_counter = 0
    for i in signs_list:
        if i in gt_sings_dict.keys() and loop_counter < len(gt_sings_dict.keys()):
            key_x = i[0]
            key_y = i[1]
            val_x = gt_sings_dict[i][0]
            val_y = gt_sings_dict[i][1]
            if int(mat[key_x][key_y]) <= int(mat[val_x][val_y]):
                wrong_bigger_counter -= 2
                opt_lst.append([[key_x, key_y], [val_x, val_y]])
        elif i in lt_sings_dict.values() and loop_counter >= len(gt_sings_dict.keys()):
            key_x = i[0]
            key_y = i[1]
            val_x = lt_sings_dict_for_comp[i][0]
            val_y = lt_sings_dict_for_comp[i][1]
            if int(mat[key_x][key_y]) >= int(mat[val_x][val_y]):
                wrong_bigger_counter -= 2
                opt_lst.append([[key_x, key_y], [val_x, val_y]])
        loop_counter +=1
    return wrong_bigger_counter, opt_lst


# this func transforms opt_lst to lists
def from_opt_to_lists(opt_lst, size):
    col_sec_cons_list = [0 for i in range(size)]
    row_sec_cons = [0 for i in range(size)]
    for item in opt_lst:
        # if the pair is in the same row
        if item[0][0] == item[1][0]:
            row_sec_cons[item[0][0]] -= 2
        # if the pair is in the same column
        elif item[0][1] == item[1][1]:
            col_sec_cons_list[item[0][1]] -= 2
    return row_sec_cons, col_sec_cons_list


# calculate the number of constraints from the <> type for both rows and columns
def constraints_calc(size):
    total_constraints = 0
    row_cons = [0 for i in range(size)]
    col_cons = [0 for i in range(size)]
    for item in gt_signs_coords_lateral.keys():
        col_cons[item[1]] += 1
        total_constraints += 1
    for item in lt_signs_coords_lateral.keys():
        col_cons[item[1]] += 1
        total_constraints += 1
    for item in gt_signs_coords_horizontal.keys():
        row_cons[item[0]] += 1
        total_constraints += 1
    for item in lt_signs_coords_horizontal.keys():
        row_cons[item[0]] += 1
        total_constraints += 1
    total_constraints += size * 2
    return row_cons, col_cons, total_constraints


# this func calculates the final matrix score
def final_score_calc(rows_score, colunm_score, row_sec_cons, col_sec_cons):
    list_size = len(rows_score)
    final_score = 0
    unfulfilled_cons = 0
    # rows
    for i in range(list_size):
        temp_score = rows_score[i] - list_size + row_sec_cons[i]
        final_score = final_score + temp_score
        if rows_score[i] != 5:
            unfulfilled_cons += 1
        if row_sec_cons[i] != 0:
            unfulfilled_cons += abs(row_sec_cons[i]) / 2
    # columns
    for i in range(list_size):
        temp_score = colunm_score[i] - list_size + col_sec_cons[i]
        final_score = final_score + temp_score
        if colunm_score[i] != 5:
            unfulfilled_cons += 1
        if col_sec_cons[i] != 0:
            unfulfilled_cons += abs(col_sec_cons[i]) / 2
    final_score = abs(final_score)
    return final_score, unfulfilled_cons


# this func fills the matrix with random numbers
def insert_rands(mat, size):
    for x in range(size):
        for y in range(size):
            if mat[x][y] == 0:
                rand_num = randint(1, size)
                mat[x][y] = rand_num


# this func preforms a mid-generation analysis
def mid_generation_analysis(matrices, size):
    maximum = MAXIMUM
    max_mat = []
    max_counter = 0
    for matrix in matrices:
        fit, counter = new_fitness(matrix, size)
        if fit < maximum or counter > max_counter:
            max_counter = counter
            maximum = fit
            max_mat = matrix
    max_counter = int(max_counter * 100)
    return maximum, max_counter, max_mat


# this func create a copy of the matrix
def copy_mat(mat, size):
    new_mat = []
    row = []
    for i in range(size):
        for j in range(size):
            row = []
            for k in range(size):
                row.append(int(mat[j][k]))
        new_mat.append(row)
    return new_mat


# this func gets a regular matrix and convert it to a binary numbers list
def get_binary_matrix(mat):
    binary_mat = []
    for row in mat:
        for cell in row:
            binary_mat.append(cell)
    return binary_mat


# this converts a binary numbers list to a regular matrix
def binary_to_regular(bin_mat, size):
    mat = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(bin_mat[size * i + j])
        mat.append(row)
    return mat


# this func prints the matrix
def print_matrix(mat, size):
    # print the board of the matrix
    y_gt = []
    y_lt = []
    space_of_a_cell = "     "  # last update -
    space_between_cells = "   "
    for x in range(size):
        for y in range(size):
            current_cell_coords = (x, y)
            # check in which list the current cell is located:
            if current_cell_coords in lt_signs_coords_horizontal.keys():
                print("[", mat[x][y], "]", end=" < ")
                if current_cell_coords in lt_signs_coords_lateral.keys():
                    y_lt.append(y)
                elif current_cell_coords in gt_signs_coords_lateral.keys():
                    y_gt.append(y)
            elif current_cell_coords in gt_signs_coords_horizontal.keys():
                print("[", mat[x][y], "]", end=" > ")
                if current_cell_coords in lt_signs_coords_lateral.keys():
                    y_lt.append(y)
                elif current_cell_coords in gt_signs_coords_lateral.keys():
                    y_gt.append(y)
            elif current_cell_coords in lt_signs_coords_lateral.keys():
                print("[", mat[x][y], "]", end="   ")
                y_lt.append(y)
            elif current_cell_coords in gt_signs_coords_lateral.keys():
                print("[", mat[x][y], "]", end="   ")
                y_gt.append(y)
            else:
                print("[", mat[x][y], "]", end="   ")
        print()  # going down a line
        while len(y_gt) != 0 or len(y_lt) != 0:
            if len(y_gt) == 0:
                # if only y_lt is not empty
                if len(y_lt) > 1:
                    print(y_lt[0] * (space_of_a_cell + space_between_cells), end="")
                    print("  V  ", end="")
                    while len(y_lt) > 1:
                        distance_between_y = y_lt[1] - y_lt[0]
                        y_lt.pop(0)
                        print(distance_between_y * space_between_cells + (distance_between_y-1) * space_of_a_cell, end="")
                        print("  V  ", end="")
                        if len(y_lt) == 1:
                            y_lt.pop(0)
                elif len(y_lt) == 1:
                    print(y_lt[0] * (space_of_a_cell + space_between_cells), end="") # busted!
                    print("  V  ", end="")
                    y_lt.pop(0)
            if len(y_lt) == 0:
                # if only y_gt is not empty
                if len(y_gt) > 1:
                    print(y_gt[0] * (space_of_a_cell + space_between_cells), end="")
                    print("  ^  ", end="")
                    while len(y_gt) > 1:
                        distance_between_y = y_gt[1] - y_gt[0]
                        y_gt.pop(0)
                        print(distance_between_y * space_between_cells + (distance_between_y-1) * space_of_a_cell, end="")
                        print("  ^  ", end="")
                        if len(y_gt) == 1:
                            y_gt.pop(0)
                elif len(y_gt) == 1:
                    print(y_gt[0] * (space_of_a_cell + space_between_cells), end="")
                    print("  ^  ", end="")
                    y_gt.pop(0)
            if len(y_gt) != 0 and len(y_lt) != 0:
            # check which sign comes first
                if y_gt[0] < y_lt[0]: # if both are not empty
                    distance_between_y = y_lt[0] - y_gt[0]
                    # then the gt is closer to the beginning of the line
                    print((y_gt[0])*(space_of_a_cell + space_between_cells), end="")
                    print("  ^  ", end="")
                    print(distance_between_y * space_between_cells + (distance_between_y-1) * space_of_a_cell, end="")
                    print("  V  ", end="")
                    y_gt.pop(0)
                    y_lt.pop(0)
                elif y_gt[0] > y_lt[0]: # if both are not empty
                    distance_between_y = y_gt[0] - y_lt[0]
                    # then the y_lt is closer to the begining of the line
                    print(y_lt[0] * (space_of_a_cell + space_between_cells), end="")
                    print("  V  ", end="")
                    # correct formula for call after cell printing of signs
                    print(distance_between_y * space_between_cells + (distance_between_y-1) * space_of_a_cell, end="")
                    print("  ^  ", end="")
                    y_gt.pop(0)
                    y_lt.pop(0)
        print()  # going down a line
        y_gt = []
        y_lt = []


# this func performs crossover between the best matrices of the current generation and returns a new matrix
def matrices_crossover(mat1, mat2, size):
    binary_mat1 = get_binary_matrix(mat1)
    binary_mat2 = get_binary_matrix(mat2)
    new_mat_bin = []
    # generate a random index for splitting the matrices
    split_index = randint(1, size ** 2 - 2)
    for i in range(split_index):
        new_mat_bin.append(binary_mat1[i])
    for i in range(split_index, size ** 2):
        new_mat_bin.append(binary_mat2[i])
    new_mat = binary_to_regular(new_mat_bin, size)
    return new_mat


# this func change the value of a random cell in the matrix
def create_mutation(matrix, size):
    # generating random x and y indices
    rand_x = randint(0, size-1)
    rand_y = randint(0, size-1)
    # making sure not to change an initial digit given in the input
    while (rand_x, rand_y) in initial_digits_dict.keys():
        rand_x = randint(0, size - 1)
        rand_y = randint(0, size - 1)
    # getting the original cell value
    origi_val = matrix[rand_x][rand_y]
    bool_get_rand = True
    # generating random vals for the chosen cell until a different cell value is generated.
    while bool_get_rand:
        rand_val = randint(1, size)
        if rand_val != origi_val:
            matrix[rand_x][rand_y] = rand_val
            bool_get_rand = False
    return matrix


# this func changes the location of the numbers that don't fit <> sings
def optimization(mat, size, switch_list,counter = 0):
    opt_mat = copy_mat(mat, size)
    # list has to be in this form: [[[x1,y1],[x2,y2]],[[,],[,]].....]
    for indices in switch_list:
        cell1 = indices[0]
        cell2 = indices[1]
        cell1_old_val = int(opt_mat[cell1[0]][cell1[1]])
        cell2_old_val = int(opt_mat[cell2[0]][cell2[1]])
        if cell1_old_val == cell2_old_val:
            if cell1_old_val == 1:
                cell1_new_val = randint(2, size)
                cell2_new_val = cell2_old_val
            elif cell1_old_val == size:
                cell1_new_val = cell1_old_val
                cell2_new_val = randint(1, size - 1)
            else:
                cell1_new_val = cell1_old_val
                cell2_new_val = randint(1, cell1_old_val-1)
        else:
            cell1_new_val = cell2_old_val
            cell2_new_val = cell1_old_val

        # check to not change the initial digits
        if (cell2[0], cell2[1]) in initial_digits_dict.keys() and (cell1[0], cell1[1]) in initial_digits_dict.keys():
            continue
        elif (cell1[0], cell1[1]) in initial_digits_dict.keys():
            # print("indices[0], indices[1]: ", indices[0], indices[1])
            if (cell2[0], cell2[1]) not in initial_digits_dict.keys():
                opt_mat[cell2[0]][cell2[1]] = cell2_new_val
        elif (cell2[0], cell2[1]) in initial_digits_dict.keys():
            # print("indices[0], indices[1]: ", indices[0], indices[1])
            if (cell1[0], cell1[1]) not in initial_digits_dict.keys():
                opt_mat[cell1[0]][cell1[1]] = cell1_new_val
        else:
            opt_mat[cell1[0]][cell1[1]] = cell1_new_val
            opt_mat[cell2[0]][cell2[1]] = cell2_new_val

    check_opt_list = []
    find_wrong_bigger_than2(opt_mat, "rows", check_opt_list)
    find_wrong_bigger_than2(opt_mat, "cols", check_opt_list)
    if len(check_opt_list) > 0:
        if counter == 10:
            return opt_mat
        optimization(opt_mat, size, check_opt_list, counter+1)
    return opt_mat


def initiate_matrices(coords_for_initial_digits, size):
    matrices = []
    for i in range(MAT_AMOUNT):
        matrix = []
        for j in range(size):
            row = []
            for k in range(size):
                if (j, k) in coords_for_initial_digits:
                    row.append(initial_digits_dict[(j, k)])
                else:
                    row.append(0)
            matrix.append(row)
        matrices.append(matrix)
    return matrices


def initiate_from_txt(argv):
    with open(argv, "r") as txt_input:
        txt_list = [(line.rstrip()) for line in txt_input]
        # get matrix size
        size = int(txt_list[0])
        # get number of initial digits
        number_of_initial_digits = int(txt_list[1])
        text_i = 2
        # fill the dict storing the initial digits
        for list_item in range(number_of_initial_digits):
            temp_item = txt_list[list_item + 2].replace(" ", "")
            x_coord = int(temp_item[0]) - 1
            y_coord = int(temp_item[1]) - 1
            xy_value = int(temp_item[2])
            complete_init_digit = [(x_coord, y_coord), xy_value]
            initial_digits_dict[(x_coord, y_coord)] = xy_value
            text_i += 1
        number_of_gt_signs = int(txt_list[text_i])
        text_i += 1
        # store the places of the gt signs
        for j in range(number_of_gt_signs):
            temp_item = txt_list[j + text_i].replace(" ", "")
            fr_x_coord = int(temp_item[0]) - 1
            fr_y_coord = int(temp_item[1]) - 1
            sec_x_coord = int(temp_item[2]) - 1
            sec_y_coord = int(temp_item[3]) - 1
            complete_location_of_greater_cell = (fr_x_coord, fr_y_coord)
            complete_location_of_lesser_cell = (sec_x_coord, sec_y_coord)
            # appending to the main dictionary: (the cell in the key is always the larger one)
            gt_signs_coords[complete_location_of_greater_cell] = complete_location_of_lesser_cell
            # print("gt coords appended: ", temp_item[0], temp_item[1], " : ", temp_item[2], temp_item[3])
            # note to self = gt == > and lt == <
            # another note - in lateral signs, the sign refers to the cell that is not checked at the moment
            # for example in the print function when a cell is in lt list as key it means that the sign below
            # it will be "V" as the cell below has lower value than the one checked
            if complete_location_of_greater_cell[0] > complete_location_of_lesser_cell[0]:
                # case 1 - lateral where the greater cell is on the bottom (x1 > x2)
                gt_signs_coords_lateral[complete_location_of_lesser_cell] = complete_location_of_greater_cell
                gt_coords_cols[complete_location_of_greater_cell] = complete_location_of_lesser_cell
            elif complete_location_of_greater_cell[0] < complete_location_of_lesser_cell[0]:
                # case 2 - lateral where the greater cell is on the top (x1 < x2)
                lt_signs_coords_lateral[complete_location_of_greater_cell] = complete_location_of_lesser_cell
                gt_coords_cols[complete_location_of_greater_cell] = complete_location_of_lesser_cell

            elif complete_location_of_greater_cell[1] > complete_location_of_lesser_cell[1]:
                # case 3 - horizontal where the greater cell is on the right (y1 > y2)
                lt_signs_coords_horizontal[complete_location_of_lesser_cell] = complete_location_of_greater_cell
                gt_coords_rows[complete_location_of_greater_cell] = complete_location_of_lesser_cell
            elif complete_location_of_greater_cell[1] < complete_location_of_lesser_cell[1]:
                # case 4 - horizontal where the greater cell is on the left (y1 < y2)
                gt_signs_coords_horizontal[complete_location_of_greater_cell] = complete_location_of_lesser_cell
                gt_coords_rows[complete_location_of_greater_cell] = complete_location_of_lesser_cell
    return size


# this func gets the fitness results for each matrix
def new_fitness(matrix, size):
    # print(matrix)
    opt_list = []
    # compute the score for the first Constraint for rows
    rows_score = rows_checker(matrix, size)
    find_wrong_bigger_than2(matrix, "rows", opt_list)
    # compute the score for the first Constraint for columns
    colunm_score = column_checker(matrix, size)
    find_wrong_bigger_than2(matrix, "columns", opt_list)
    # compute the score for the second constraint for the ros and columns:
    row_sec_cons, col_sec_cons = from_opt_to_lists(opt_list, size)
    # compute the number of <> constraints per row,column
    row_tot_cons, col_tot_cons, total_cons = constraints_calc(size)
    # compute the total score of the matrix
    final_mat_score, unfulfilled_cons = final_score_calc(rows_score, colunm_score, row_sec_cons, col_sec_cons)
    # compute number of constraints fulfilled to not fulfilled ratio:
    fulfilled_cons = total_cons - unfulfilled_cons
    ratio = fulfilled_cons / total_cons
    return final_mat_score, ratio


# this func gets all the generation matrices and gets their fitness from the fitness func
def new_generations(generation, algorithm, size):
    fitness_list = []
    list_for_opt = []
    gen_res = []
    fit_and_mat_lst = []
    sorted_mats = []
    # calculate fitness according to the given algorithm
    if algorithm == "Normal":
        for i in range(0, MAT_AMOUNT):
            fitness_list.append(new_fitness(generation[i], size)[0])
    if algorithm == "Darwin":
        for i in range(0, MAT_AMOUNT):
            find_wrong_bigger_than2(generation[i], "rows", list_for_opt)
            find_wrong_bigger_than2(generation[i], "cols", list_for_opt)
            opt_res = optimization(generation[i], size, list_for_opt)
            fitness_list.append(new_fitness(opt_res, size)[0])
    if algorithm == "Lamarck":
        for i in range(0, MAT_AMOUNT):
            find_wrong_bigger_than2(generation[i], "rows", list_for_opt)
            find_wrong_bigger_than2(generation[i], "cols", list_for_opt)
            opt_res = optimization(generation[i], size, list_for_opt)
            opt_fit = new_fitness(opt_res, size)[0]
            old_fit = new_fitness(generation[i], size)[0]
            if opt_fit < old_fit:
                generation[i] = opt_res
            fitness_list.append(new_fitness(generation[i], size)[0])
    # appending all mats with their fitness value
    for j in range(0, MAT_AMOUNT):
        fit_and_mat_lst.append([fitness_list[j], generation[j]])
    # sorting the fit_and_mat_lst list and replicating the best couple
    fit_and_mat_sorted = sorted(fit_and_mat_lst, reverse=False)
    for k in range(MAT_AMOUNT):
        sorted_mats.append(fit_and_mat_sorted[k][1])
    for w in range(2):
        for j in range(REP_NUM):  # replicating 5 times
            gen_res.append(sorted_mats[w])
    # doing crossover
    for n in range(CROSS_NUM):
        random_mat1 = randint(0, MAT_AMOUNT - 1)
        random_mat2 = randint(0, MAT_AMOUNT - 1)
        gen_res.append(matrices_crossover(sorted_mats[random_mat1], sorted_mats[random_mat2], size))
    # creating mutations
    for m in range(MAT_AMOUNT):
        prob = random()
        if prob < MAX_PROB:
            gen_res[m] = create_mutation(gen_res[m], size)
    return gen_res


if __name__ == "__main__":
    #seed(10)
    # receiving values from the input file
    size = initiate_from_txt(sys.argv[1])
    print("The matrix size is: ", size, "*", size)
    # list of points to fill with init digits:
    coords_for_digits = [point for point in initial_digits_dict.keys()]
    matrices = initiate_matrices(coords_for_digits, size)

    # get algorithm
    print("Please Choose one of the following algorithms: Normal, Darwin, Lamarck")
    algo = input()
    print("The Chosen algorithm: ", algo)
    print("Number of generations: ", GENS_NUM)
    print("Number of metrics in each generation: ", MAT_AMOUNT)

    # inserting numbers into the generation matrices
    for matrix in matrices:
        insert_rands(matrix, size)
    gens_num = GENS_NUM
    for i in range(gens_num):
        matrices = new_generations(matrices, algo, size)
        if i % 100 == 0:
            mid_fit, mid_ratio, optimal_mat = mid_generation_analysis(matrices, size)
            if mid_ratio == 100:
                break

    # find the highest fitness
    max_mat = []
    max_counter = 0
    maximum = MAXIMUM
    for matrix in matrices:
        fit, counter = new_fitness(matrix, size)
        if fit < maximum or counter > max_counter:
            max_counter = counter
            maximum = fit
            max_mat = matrix
    max_counter = int(max_counter * 100)

    print("\nmaximum fitness ratio: ", str(max_counter) + "%")
    print("maximum fitness score: ", maximum)
    print_matrix(max_mat, size)

