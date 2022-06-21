# Dafna Shlomi 302362173, Itay Sova 313246209

import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import matplotlib as mpl
from hexalattice.hexalattice import *

INPUT_FILE = "Elec_24.csv"
MAP_DIM = 8
COLS_NUM = 13
HEXS_NUM = 61
DATA_ROWS = 196
rand = np.random.RandomState()
EPOCHS = 30
SHUFFLE = True
REPEATS = 10
w = [0.6, 0.4, 0.2, 0.05, 0.01]
WIGHTS = np.array([item for item in w])
LEARNING = 0.1
# dict of vectors and centers
dict_vec_center = {}
# dict of vectors and their economic value
dict_vec_eco = {}
# dict of vectors and their municipally
dict_vec_mun = {}
# dict of vectors and their economic value color
dict_vec_color = {}

# defining a color for each economic value
c1 = [0.0, 0.0, 1.0]  # blue = 1
c2 = [0.0, 0.5, 1]  # 2
c3 = [0.0, 1, 1]  # cyan = 3
c4 = [0.0, 1, 0.5]  # 4
c5 = [0.5, 1.0, 0.0]  # 5
c6 = [1.0, 1.0, 0.0]  # yellow = 6
c7 = [1, 0.5, 0.0]  # orange = 7
c8 = [1, 0.0, 0.0]  # red = 8
c9 = [1, 0.0, 0.5]  # 9
c10 = [1, 0.0, 1.]  # pink 10
arr = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]
dict_colors = {'1': c1, '2': c2, '3': c3, '4': c4, '5': c5, '6': c6, '7': c7, '8': c8, '9': c9, '10': c10}


# this func gets the municipalities of each center
def get_final_res_no_print():
    centers_cities_list = [[i, []] for i in range(len(hex_centers))]
    for vec, cen in dict_vec_center.items():
        centers_cities_list[cen][1].append(dict_vec_mun[vec])
    new_centers_cities_list = [cen for cen in centers_cities_list if len(cen[1]) != 0]
    res_list =[(centers_list[i[0]],i[1]) for i in new_centers_cities_list]
    return res_list


# this func prints the municipalities of each center
def print_final_ans(res_list):
    for i in res_list:
        print("\nThe Center Coordinates: ", i[0])
        print("Municipalities In Center: ", i[1])


# this function receives a vector and updates the weights of the Soms and colors of its neighboring cells
def update_weights(SOM, input_vec, nig_list):
    # receiving the vectors color
    train_colour = dict_vec_color[str(input_vec)]
    # for every level of neighbors and wight
    for n, w in zip(nig_list, WIGHTS):
        # updating the values of every neighbor in the level
        for i in n:
            SOM[i] += LEARNING * np.array(w) * (input_vec - SOM[i])
            fixed_colors[i] += LEARNING * np.array(w) * (train_colour - fixed_colors[i])
    return SOM


# this func returns the index of the center with the closet som values to the vector x
def find_BMU(SOM, x):
    distSq = (np.square(SOM - x)).sum(axis=1)
    return np.argmin(distSq, axis=None)


# plotting the grid
def plot_som(centers, colors, init=False):
    # saving all the x vals of the centers and y vals of the centers into two separate arrays
    x_hex_coords = centers[:, 0]
    y_hex_coords = centers[:, 1]
    plot_single_lattice_custom_colors(x_hex_coords, y_hex_coords,
                                            face_color=colors,
                                            edge_color=colors * 0,
                                            min_diam=0.9,
                                            plotting_gap=0,
                                            rotate_deg=0)
    if init is False:
        norm = mpl.colors.Normalize(vmin=1, vmax=10)
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm,  cmap=plt.cm.jet), orientation='horizontal', label='Economic Scale')


# reading the input data from the input file
def get_input_vals(input_file):
    # reading the data
    data = pd.read_csv(input_file)
    # saving the columns we don't use in the compression and removing them from the data
    economic_vals = data['Economic Cluster'].values
    municipality_vals = data['Municipality'].values
    data = data.drop(columns="Economic Cluster")
    data = data.drop(columns="Municipality")
    data = data.drop(columns="Total Votes")
    input_vals = data.values
    # turning the data values into percents
    data_sums = np.sum(data, axis=1)
    for i in range(DATA_ROWS):
        for j in range(COLS_NUM):
            input_vals[i, j] = np.array(round((input_vals[i, j] / data_sums[i]) * 100))
    # creating an input vectors dicts of their Municipality, economic_value, and economic_value color
    for line, m, e in zip(input_vals, municipality_vals, economic_vals):
        dict_vec_eco[str(line)] = e
        dict_vec_mun[str(line)] = m
        dict_vec_color[str(line)] = np.array([i for i in dict_colors[str(e)]]).astype(dtype="float64")
    return input_vals


# this func receives the chosen center and returns its neighbors
def get_neighbors(cell, max_level=len(WIGHTS)):
    # the radios*2 value of the grid
    R2 = abs(abs(hex_centers[0][0]) - abs(hex_centers[1][0]))
    visited_list = []
    # creating an empty list of neighbors
    neigh_in_levels = []
    for lvl in range(max_level):
        neigh_in_levels.append([])
    # for each neighbor's level
    for j in range(0, max_level):
        # for each center
        for i in range(len(centers_list)):
            # if the cells aren't already in neighbors list, check their x and y values distance form the chosen center
            if i not in visited_list:
                x_dist = abs(np.array(centers_list[i][0] - hex_centers[cell][0]))
                y_dist = abs(np.array(centers_list[i][1] - hex_centers[cell][1]))
                sum_dists = x_dist + y_dist
                # if the x and y distances smaller or equal to the radios*2 of the level add them to the list
                if (x_dist == 0 and y_dist <= R2 * j) or (y_dist == 0 and x_dist <= R2 * j) \
                        or (x_dist < R2 * j and y_dist < R2 * j and sum_dists <= R2 + j + 0.5):
                    visited_list.append(i)
                    neigh_in_levels[j].append(i)
    return neigh_in_levels


# this func calculates the quantization error of each final result grid
def evaluate_solution(SOM, input_vectors):
    list_of_dist_between_input_and_center = []
    for item in input_vectors:
        distSq = (np.square(SOM - item)).sum(axis=1)
        min_dist_curr = np.sqrt(np.amin(distSq))
        list_of_dist_between_input_and_center.append(min_dist_curr)
    avg_dist_between_input_and_center = np.average(np.array(list_of_dist_between_input_and_center))
    return avg_dist_between_input_and_center


if __name__ == '__main__':
    # receiving the input vectors
    file_name = sys.argv[1]
    input_vectors = get_input_vals(file_name)
    print("\nGenerating 10 different SOMs of the data file: ", file_name, ", please wait.")

    # creating a grid of hexs
    hex_centers, _ = create_hex_grid(n=100, crop_circ=4, do_plot=True)
    plt.close('all')
    # initializing lists of hex centers
    centers_list = (hex_centers.copy()).tolist()

    repeats, shuffle = REPEATS, SHUFFLE
    list_of_eval = []
    list_of_maps = []
    list_of_init_maps = []
    res_list = []
    for r in range(repeats):
        # creating randomized som vectors in range between 0 and 100
        soms_vecs = rand.randint(0., 100., (HEXS_NUM, COLS_NUM))
        fixed_som = np.array([item for item in soms_vecs]).astype(dtype="float64")

        # initializing the som grid with random colors
        colors = rand.uniform(0.0, 1.0, (HEXS_NUM, 3)).astype(float)
        fixed_colors = np.array([item for item in colors]).astype(dtype="float64")
        list_of_init_maps.append((hex_centers.copy(), fixed_colors.copy()))

        # running the epochs
        for i in range(EPOCHS):
            list_vectors_for_center = [[] for number in range(61)]
            if shuffle:
                rand.shuffle(input_vectors)
            # for each input vector: find the closet center, update chosen center in the dictionary,
            # and update the neighboring soms wights
            for vector in input_vectors:
                center = find_BMU(fixed_som, vector)
                list_vectors_for_center[center].append(vector)
                dict_vec_center.update({str(vector.copy()): center})
                fixed_som = update_weights(fixed_som, vector, get_neighbors(center))
        # saving the last result
        res_list.append(get_final_res_no_print())
        list_of_maps.append((hex_centers, fixed_colors))
        # evaluate by avg distance from input vector to center
        list_of_eval.append(evaluate_solution(fixed_som, input_vectors))

    min_mark = np.argmin(np.array(list_of_eval))
    hex_centers = list_of_maps[min_mark][0]
    fixed_colors = list_of_maps[min_mark][1]
    # showing the grids initial state
    plot_som(list_of_init_maps[min_mark][0], list_of_init_maps[min_mark][1], True)
    centers_list = (hex_centers.copy()).tolist()
    print("\nThe final results of the best generated som:")
    print_final_ans(res_list[min_mark])
    # showing the final grid
    plot_som(list_of_maps[min_mark][0], list_of_maps[min_mark][1])
    plt.show()
    exit(1)
