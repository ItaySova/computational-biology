# authors: Itay Sova, Dafna Shlomi

from tkinter import *
from random import *
from copy import deepcopy
from time import sleep
import matplotlib.pyplot as plt
from math import floor, log10

# GLOBAL LISTS
# assigning lists to save the number of sick, healthy and vaccinated - for analysis purposes
SICK_COUNT_LIST = []
HEALTHY_COUNT_LIST = []
VACCINATED_COUNT_LIST = []
# assigning a list of the matrix cells
CELLS_LIST = []
# assigning a list of matrix colours
colours = []
# assigning lists to get vals from start gui and
start_entry_list = []
start_vals = []
# index list: 0 = n_ppl, 1 = n_gens, 2 = r_percent, 3 = sick_percent, 4 = sick_gens, 5 = T_percent,
# 6 = low_p, 7 = high_p
START_LABELS = ["Number Of People: ", "Number Of Generations: ", "Percent Of Fast Moving: ",
                "Percent Of Sick: ", "Infection duration: ", "Low Spread To High Spread Threshold: ",
                "Low Spread infection Prob: ", "High Spread infection Prob: "]
END_LABELS = ["Percent Of Healthy: ", "Percent Of Vaccinated: ", "Percent Of Sick: "]
TITLES = ["The Simulation's Parameters", "The Final Results"]
# assigning empty list for R group
R = []
float_i_list2 = [2, 3, 5]

# GLOBAL VAR AND CONSTANTS
# the default size of each cell in the matrix
CELL_SIZE = 4
# the default height and width of the matrix
MAT_SIZE = 200
# setting a random percent of sick people
# RAND_S_PERCENT = 0.01
RAND_S_PERCENT = uniform(0.00, 1.00)
RAND_S_PERCENT = float(round(RAND_S_PERCENT, (-floor(log10(RAND_S_PERCENT))+1)))

# setting a dictionary of the default values of start window
START_DICT = {"Number Of People: ": "16000", "Number Of Generations: ": "20", "Percent Of Fast Moving: ": "0.05",
              "Percent Of Sick: ": str(RAND_S_PERCENT), "Infection duration: ": "5",
              "Low Spread To High Spread Threshold: ": "0.05",
              "Low Spread infection Prob: ": "0.1", "High Spread infection Prob: ": "0.5"}


# this func creates part of the validation func error messages
def error_msg(i_list):
    msg = "the values of - "
    for i in i_list:
        msg += START_LABELS[i] + ", "
    new_msg = msg[:-2]
    return new_msg


# this func validates the input received in the start window
def validation():
    int_i_list = [0, 1, 4]
    float_i_list1 = [2, 3, 5, 6, 7]
    msg_1 = error_msg(int_i_list) + " - have to be integers >= 0."
    msg_2 = error_msg([0]) + " - has to be bigger than 0 and smaller than " + str(MAT_SIZE*MAT_SIZE)
    msg_3 = error_msg(float_i_list1) + " - have to be floats between 0.0 to 1.0"
    msg_4 = "the values of " + START_LABELS[7] + " has to be bigger or equal to the value of " + START_LABELS[6]
    # making sure n_ppl, n_gens, and sick_gen are ints
    for i in int_i_list:
        assert start_vals[i].isnumeric and int(start_vals[i]) >= 0, msg_1
    # making sure number of ppl is bigger than 0 and smaller than the matrix
    assert 0 < int(start_vals[0]) <= int(MAT_SIZE*MAT_SIZE), msg_2
    # making sure R_percent, sick_percent, T_percent, high_p and low_p are floats
    for i in float_i_list1:
        assert 0 <= float(start_vals[i]) <= 1.0, msg_3
    # making sure high_p => low_p
    assert float(start_vals[6]) <= float(start_vals[7]), msg_4


# this func counts the number of healthy, vaccinated and sick in each generation
def matrix_counter():
    healthy_count = 0
    vaccinated_count = 0
    sick_count = 0
    # going over the matrix and counting the values representing the different types
    for x in range(MAT_SIZE):
        for y in range(MAT_SIZE):
            if matrix[x][y] == healthy_mat_val:
                healthy_count += 1
            if vaccinated_mat_val < matrix[x][y] <= sick_gens:
                sick_count += 1
            if matrix[x][y] == vaccinated_mat_val:
                vaccinated_count += 1
    # updating global lists
    SICK_COUNT_LIST.append(sick_count)
    HEALTHY_COUNT_LIST.append(healthy_count)
    VACCINATED_COUNT_LIST.append(vaccinated_count)
    return healthy_count, vaccinated_count, sick_count


# line graph showing the amount of people in different groups in each generation
def line_graph():
    plt.xlim(0, n_gens)
    y1 = SICK_COUNT_LIST
    plt.plot(y1, label="sick count")

    y2 = HEALTHY_COUNT_LIST
    plt.plot(y2, label="healthy count")

    y3 = VACCINATED_COUNT_LIST
    plt.plot(y3, label="vaccinated count")

    plt.legend()
    plt.ylabel('people count')
    plt.xlabel('generation')
    plt.title('Sick, Healthy and Vaccinated count per generation')
    plt.show()


# func for just sick count graph
def sick_line_graph():
    plt.xlim(0, n_gens)
    y1 = SICK_COUNT_LIST
    plt.plot(y1, label="sick count")

    y2 = [int(T_percent * n_ppl) for index in range(0, n_gens)]
    plt.plot(y2, label="threshold")

    plt.ylabel('people count')
    plt.xlabel('generation')
    plt.title('Sick count per generation')
    plt.show()


# func to compute the value of X or Y in case of not fitting in matrix size
def warp_around_fix(x):
    if x >= MAT_SIZE:
        return x - MAT_SIZE
    elif x < 0:
        return MAT_SIZE + x
    else:
        return x


# init R group
def R_group_init(mat, size_of_R):
    genetal_occ_locations = []
    # going through the matrix and looking for occupied cells
    for x in range(MAT_SIZE):
        for y in range(MAT_SIZE):
            if mat[x][y] != empty_mat_val:
                genetal_occ_locations.append((x, y))
    list_of_indexes = [index for index in range(0, len(genetal_occ_locations))]
    shuffle(list_of_indexes)
    indexes_of_R = [list_of_indexes[index] for index in range(0, size_of_R)]
    final_loc_r = []
    for index in indexes_of_R:
        final_loc_r.append(genetal_occ_locations[index])
    return final_loc_r


# R group move func
def R_group_move(mat, new_mat, r, c):
    cell_neighbors = []
    cell_neighbors.append((r, c))
    # after appending the current cell go through the cell and its neighbors
    for i in range(r - 10, r + 11, 10):
        for j in range(c - 10, c + 11, 10):
            # making two temp variables for warp_around
            temp_i = warp_around_fix(i)
            temp_j = warp_around_fix(j)
            # if there are empty neighboring cells - add them to the list
            if new_mat[temp_i][temp_j] == empty_mat_val and mat[temp_i][temp_j] == empty_mat_val:
                # connecting the full matrix and the cell's neighborhood
                if (temp_i, temp_j) not in cell_neighbors:
                    cell_neighbors.append((temp_i, temp_j))
    # choose an empty cell to move to or stay in place
    new_spot = choice(cell_neighbors)
    # get the cell location (row,col) and update the matrix
    cell_row = int(new_spot[0])
    cell_col = int(new_spot[1])
    new_mat[cell_row % MAT_SIZE][cell_col % MAT_SIZE] = mat[r][c]
    R.remove((r, c))
    R.append((cell_row, cell_col))


# this func update the current_mat cells colour
def update_mat_colours(current_mat, current_can):
    for row in range(MAT_SIZE):
        for col in range(MAT_SIZE):
            for val in mat_vals_range:
                if current_mat[row][col] == val:
                    current_can.itemconfig(CELLS_LIST[row][col], fill=colours[val])


# this func helps to initialize the matrix with sick and healthy people
def init_sick_and_healthy(n_ppl_type, taken, m_val):
    # choosing a random cell from taken list, getting its row and column and updating the matrix
    for i in range(n_ppl_type):
        cell = choice(taken)
        cell_row = cell % MAT_SIZE
        cell_col = int(cell / MAT_SIZE)
        taken.remove(cell)
        matrix[cell_row][cell_col] = m_val
    return taken


# this func creates a new matrix with sick and healthy cells in random locations
def init_mat_cells(sick_num, healthy_num, canvas):
    taken_cell = list(range(MAT_SIZE ** 2))
    taken_cell = init_sick_and_healthy(sick_num, taken_cell, sick_mat_val)
    init_sick_and_healthy(healthy_num, taken_cell, healthy_mat_val)
    # making the cells and the boundaries of the matrix
    for j in range(MAT_SIZE):
        mat_r = []
        for i in range(MAT_SIZE):
            i_beginning = i * CELL_SIZE + 1
            j_beginning = j * CELL_SIZE + 1
            i_end = i * CELL_SIZE + CELL_SIZE + 3
            j_end = j * CELL_SIZE + CELL_SIZE + 3
            mat_r.append(canvas.create_rectangle(i_beginning, j_beginning, i_end, j_end, fill='black'))


# TWO PEOPLE IN THE SAME CELL PREVENTION - SUB FUNC
# this func returns a list of optional cell to move to
def get_empty_neighbors_list(row, col, new_mat, mat):
    neighbors_list = [4]
    # after appending the current cell go through the cell and its neighbors
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            # if there are empty neighboring cells - add them to the list
            p = new_mat[i % MAT_SIZE][j % MAT_SIZE] == empty_mat_val
            q = mat[i % MAT_SIZE][j % MAT_SIZE] == empty_mat_val
            if p and q:
                neighbors_list.append((i - row + 1) * 3 + j - col + 1)
    return neighbors_list


# TWO PEOPLE IN THE SAME CELL PREVENTION - MAIN FUNC #
# this func go though the current matrix and a new copy of it,
# when it finds a cell with a person in - the function calls a sub function that goes though the
# neighboring cells and saves the empty cells that are empty in both current matrix and new matrix
# and than moves the person (or doesnt move) to one of the optional cells at random.
# after the move the new matrix get updated.
def locations_update(mat):
    # creating a matrix and going through its cells
    mat_new = [[empty_mat_val for x in range(MAT_SIZE)] for y in range(MAT_SIZE)]
    for row in range(MAT_SIZE):
        for col in range(MAT_SIZE):
            # if the current cell isn't empty (there is a person in it), get a list of close cells
            if mat[row][col] != empty_mat_val:
                if (row, col) not in R:
                    neighbors_list = get_empty_neighbors_list(row, col, mat_new, mat)
                    # choosing a neighboring cell to move to at random and saving its row and column
                    new_cell = choice(neighbors_list)
                    cell = new_cell
                    cell_row = int(cell / 3)
                    cell_col = cell % 3
                    # updating the new matrix
                    mat_new[(cell_row + row - 1) % MAT_SIZE][(cell_col + col - 1) % MAT_SIZE] = mat[row][col]
                elif (row, col) in R:
                    R_group_move(mat, mat_new, row, col)
    return mat_new


# this func checks if a person will get infected by its neighbors and updates the matrix
def infection_update(mat, is_high_p):
    # creating a copy of the matrix and going through its cells
    mat_copy = deepcopy(mat)
    for row in range(MAT_SIZE):
        for col in range(MAT_SIZE):
            # if there is a healthy person in the cell
            if mat[row][col] == healthy_mat_val:
                # checking the adjacent cells for sick neighbors
                for x in range(row - 1, row + 2):
                    for y in range(col - 1, col + 2):
                        if vaccinated_mat_val < mat[x % MAT_SIZE][y % MAT_SIZE] <= sick_gens:
                            # IF VIRUS SPREAD IS LOW - infection prob is high
                            if is_high_p and random() < high_p:
                                mat_copy[row][col] = sick_mat_val
                            # IF VIRUS SPREAD IS HIGH - infection prob is low
                            elif not is_high_p and random() < low_p:
                                mat_copy[row][col] = sick_mat_val
            # if a cell has a sick person in it - remove one gen of sickness from its count
            if 0 < mat_copy[row][col] <= sick_gens:
                mat_copy[row][col] -= 1
    return mat_copy


# this func creates the final results window labels
def create_res_gui_lbls(lbls, frame, info):
    for i in range(len(lbls)):
        Label(frame, text=lbls[i] + str(info[i]),
              font="Calibri 14", justify=LEFT, padx=3, pady=3).grid(row=i, column=0, sticky=NW)


# this func creates the label frames inside the roots of start and results window
def create_lbl_frames(lbls,frame, r, c):
    lf = LabelFrame(frame, text=lbls, font="Calibri 16 underline bold", padx=10, pady=10
                    , borderwidth=0, highlightthickness=0)
    lf.grid(row=r, column=c,  sticky="WN", padx=10, pady=10)
    return lf


# this func creates the gui of the final results window
def create_res_gui(para_vals, final_count):
    res_root = Tk()
    res_root.title("Covid Spread Simulation - Final Results")

    # showing the final results
    res_frame = create_lbl_frames(TITLES[1], res_root, 0, 1)
    final_count_percent = [0, 0, 0]
    # turning numbers to percents
    for i in range(len(final_count)):
        res_percent = float(final_count[i] / n_ppl)
        final_count_percent[i] = float(round(res_percent, 5))

    # showing the parameters
    create_res_gui_lbls(END_LABELS, res_frame, final_count_percent)
    para_frame = create_lbl_frames(TITLES[0], res_root, 2, 1)
    create_res_gui_lbls(START_LABELS, para_frame, para_vals)

    # creating the exit button
    okVar = IntVar()
    end_b = Button(para_frame, text="Exit", font="Calibri 14", padx=5, pady=5, command=lambda: okVar.set(1))
    end_b.grid(row=9, padx=10, pady=10)
    res_root.update()
    return res_root, end_b, okVar


# this func saves the values inserted in the start window
def after_start_click():
    for i in range(len(start_entry_list)):
        val = start_entry_list[i].get()
        if i in float_i_list2 and float(val) < float(1/int(start_vals[0])):
            val = 0
        START_DICT[START_LABELS[i]] = val
        start_vals.append(str(val))


# this func creates the gui the Simulation
def create_simulation_gui():
    simulation_root = Tk()
    simulation_root.title('Covid Spread Simulation')
    # creating a matrix gui
    matrix_canvas = Canvas(simulation_root, bd=0, height=MAT_SIZE * CELL_SIZE, width=MAT_SIZE * CELL_SIZE)
    matrix_canvas.config(bg='black')
    matrix_canvas.pack()
    for r in range(MAT_SIZE):
        rows_list = []
        for k in range(1, MAT_SIZE + 1):
            rows_list.append(r * MAT_SIZE + k)
        CELLS_LIST.append(rows_list)
    return matrix_canvas, simulation_root


# this func creates the gui of start window
def create_start_gui():
    start_root = Tk()
    start_root.title("Covid Spread Simulation - Start Menu")
    start_frame = create_lbl_frames(TITLES[0], start_root, 0, 1)
    # creating the start windows entries with the default values
    for i in range(len(START_LABELS)):
        Label(start_frame, text=START_LABELS[i], font="Calibri 14", padx=3, pady=3).grid(row=i, column=0, sticky=NW)
        e = Entry(start_frame, width=6, font="Calibri 14")
        e.insert(i, START_DICT[START_LABELS[i]])
        e.grid(row=i, column=1)
        start_entry_list.append(e)
    # creating the start button
    okVar = IntVar()
    start_b = Button(start_frame, text="Start", font="Calibri 14", padx=5, pady=5, command=lambda: okVar.set(1))
    start_b.grid(row=8, padx=10, pady=10)
    return start_root, okVar


# this func creates the matrix and paint the cells according to their content:
# vaccinated person = Cyan, sick person = Red, healthy person = Yellow, empty cell = Black
def mat_init():
    new_matrix = [[empty_mat_val for r in range(MAT_SIZE)] for c in range(MAT_SIZE)]
    colours.append("Cyan")
    for i in range(sick_gens):
        colours.append("Red")
    colours.append("Black")
    colours.append("Yellow")
    return new_matrix


# the main func runs the programs functions
if __name__ == '__main__':
    # creating the start window gui
    s_root, ok = create_start_gui()
    # closing the start window getting its values
    s_root.wait_variable(ok)
    after_start_click()
    validation()
    s_root.destroy()

    # assigning the values of the program variables using values from the start window
    n_ppl = int(START_DICT[START_LABELS[0]])
    n_gens = int(START_DICT[START_LABELS[1]])
    R_percent = float(START_DICT[START_LABELS[2]])
    size_of_R = int(n_ppl * R_percent)
    sick_percent = float(START_DICT[START_LABELS[3]])
    n_sick = int(sick_percent * n_ppl)
    n_healthy = n_ppl - n_sick
    sick_gens = int(START_DICT[START_LABELS[4]])
    T_percent = float(START_DICT[START_LABELS[5]])
    low_p = float(START_DICT[START_LABELS[6]])
    high_p = float(START_DICT[START_LABELS[7]])

    # assigning numbers to represent the different cell states on the matrix
    vaccinated_mat_val = 0
    sick_mat_val = sick_gens
    empty_mat_val = sick_gens + 1
    healthy_mat_val = sick_gens + 2
    mat_vals_count = sick_gens + 3
    mat_vals_range = list(range(mat_vals_count))

    # creating the matrix
    matrix = mat_init()

    # creating the simulation gui and running the model
    can, sim_root = create_simulation_gui()
    init_mat_cells(n_sick, n_healthy, can)
    R = R_group_init(matrix, size_of_R)
    update_mat_colours(matrix, can)
    sim_root.update()
    sleep(0.1)

    # while the number of generations is smaller than the limit run the simulation
    gen_counter = 0
    while gen_counter < n_gens:
        sick_count = matrix_counter()[2]
        # checking if the spread of the virus (the percent of sick individuals) is bigger or lower than T_percent
        if float(sick_count / n_ppl) > T_percent:
            bool_high_p = False
        else:
            bool_high_p = True
        # updating the people locations and health m_val
        matrix = infection_update(matrix, bool_high_p)
        matrix = locations_update(matrix)
        update_mat_colours(matrix, can)
        gen_counter += 1
        sim_root.update()
        sleep(0.1)

    # creating a plot showing the spread of the virus though the generations
    if n_gens > 0:
        sick_line_graph()
        line_graph()
    # receiving vals for the final results screen
    final_count = matrix_counter()
    # closing the simulation gui
    sim_root.destroy()
    # showing the final results gui
    e_root, b_roo, ok = create_res_gui(start_vals, final_count)
    # close the final results gui after clicking the 'exit' button
    e_root.wait_variable(ok)
    e_root.destroy()

