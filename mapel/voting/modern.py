import matplotlib.pyplot as plt
import numpy as np
from . import objects as obj
import os


def print_2d(name, num_winners=0, shades=False):
    model = obj.Model_2d(name, num_winners=num_winners, shades=shades)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if shades:
        file_name = "results/times/" + str(name) + ".txt"
        file_ = open(file_name, 'r')
        ctr = 0
        for k in range(model.num_families):
            for _ in range(model.family_size[k]):
                if ctr < 200:
                    shade = float(file_.readline())
                else:
                    shade = 1.
                ax.scatter(model.points[ctr][0], model.points[ctr][1],
                           color=model.colors[k], alpha=model.alphas[k]*shade, s=9)
                ctr += 1

    else:
        #for w in model.winners:
        #for w in [513, 527]:
        #    ax.scatter(model.points[w][0], model.points[w][1], color="black", s=30)
        for k in range(model.num_families):
            ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                       color=model.colors[k], label=model.labels[k],
                       alpha=model.alphas[k], s=9)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    file_name = os.path.join(file_name, "images", str(name) + ".png")
    plt.savefig(file_name)
    plt.show()
    
 
def generate_matrix(experiment_id):
    # generate MATRIX

    experiment_id = "example_100_3"

    ######
    file_controllers = open("experiments/" + experiment_id + "/controllers/example_100_3.txt", 'r')
    num_families = int(file_controllers.readline())
    num_families = int(file_controllers.readline())
    num_families = int(file_controllers.readline())
    family_name = [0 for _ in range(num_families)]
    family_special = [0 for _ in range(num_families)]
    family_size = [0 for _ in range(num_families)]
    family_full_name = [0 for _ in range(num_families)]

    labels = [0 for _ in range(num_families)]
    colors = [0 for _ in range(num_families)]
    alphas = [0 for _ in range(num_families)]

    for i in range(num_families):
        line = file_controllers.readline().rstrip("\n").split(',')

        family_name[i] = str(line[1])
        family_special[i] = float(line[2])
        family_size[i] = int(line[0])
        family_full_name[i] = str(line[5])

        labels[i] = str(line[1])
        colors[i] = str(line[3])
        alphas[i] = float(line[4])
    ######

    file_name = "experiments/" + str(experiment_id) + "/results/distances/example_100_3.txt"
    file_ = open(file_name, 'r')

    num_elections = int(file_.readline())
    second_line = int(file_.readline())
    third_line = int(file_.readline())

    matrix = [[0. for _ in range(second_line)] for _ in range(second_line)]
    quan = [[0 for _ in range(second_line)] for _ in range(second_line)]

    code = [0 for _ in range(num_elections)]

    ctr = 0
    for i in range(num_families):
        for j in range(family_size[i]):
            code[ctr] = i
            ctr += 1

    #print(code)

    for i in range(third_line):
        line = file_.readline().split(' ')
        # print(code[int(line[0])])
        a = code[int(line[0])]
        b = code[int(line[1])]
        value = float(line[2])
        matrix[a][b] += value
        quan[a][b] += 1

    for i in range(second_line):
        for j in range(i, second_line):
            matrix[i][j] /= float(quan[i][j])
            matrix[i][j] *= 10.
            matrix[i][j] = int(round(matrix[i][j], 0))
            matrix[j][i] = matrix[i][j]

    #print(matrix[0][0])

    print("ok")
    file_.close()

    fig, ax = plt.subplots()

    min_val, max_val = 0, num_families

    intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))

    #print(intersection_matrix)

    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i in xrange(num_families):
        for j in xrange(num_families):
            c = int(matrix[i][j])
            ax.text(i, j, str(c), va='center', ha='center')



    x_values = family_full_name
    y_values = family_full_name
    y_axis = np.arange(0, num_families, 1)
    x_axis = np.arange(0, num_families, 1)

    # plt.barh(y_axis, x_values, align='center')
    plt.yticks(y_axis, y_values)
    plt.xticks(x_axis, x_values, rotation='vertical')

    plt.savefig("matrix_luty.png")
    plt.show()



