import matplotlib.pyplot as plt
import numpy as np
from . import objects as obj
import os


def print_2d(name, num_winners=0, shades=False, mask=False, angle=0, reverse=False, update=False):

    model = obj.Model_2d(name, num_winners=num_winners, shades=shades)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # rotate by angle
    if angle != 0:
        model.rotate(angle)

    if reverse:
        model.reverse()

    if update:
        model.update()

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

        for k in range(model.num_families):
            if k < 30:
                ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                           color=model.colors[k], label=model.labels[k],
                           alpha=model.alphas[k], s=9)
            else:
                ax.scatter(model.points_by_families[k][0], model.points_by_families[k][1],
                           color=model.colors[k], label=model.labels[k],
                           alpha=model.alphas[k], s=30)

        for w in model.winners:
            ax.scatter(model.points[w][0], model.points[w][1], color="black", s=20)


    if mask:

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis('off')
        plt.margins(0.25)
        name = "images/" + str(name) + ".png"
        plt.savefig(name)
        #plt.show()

        print("mask")

        background = Image.open("images/example_100_100.png")
        c = 0.59
        x = int(1066*c)
        y = int(675*c)
        foreground = Image.open("images/mask.png").resize((x, y))

        background.paste(foreground, (-30, 39), foreground)
        background.show()
        background = background.save("geeks.png")
        name = "images/" + str(name) + ".png"
        plt.savefig(name)

    else:

        box = ax.get_position()

        #ax.set_position([box.x0, box.y0, box.width * 1, 1*box.height])
        lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis('off')

        #####################################################
        file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
        file_name = os.path.join(file_name, "experiments", str(name), "controllers", str(name) + ".txt")
        file_ = open(file_name, 'r')
        num_voters = int(file_.readline())
        num_candidates = int(file_.readline())
        text_name = "100x" + str(num_candidates)
        #####################################################

        text = ax.text(0.0, 1.05, text_name, transform=ax.transAxes)
        #name = "images/" + str(name) + ".png"

        file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
        file_name = os.path.join(file_name, "images", str(name) + "_map.png")

        plt.savefig(file_name, bbox_extra_artists=(lgd, text), bbox_inches='tight')
        plt.show()

        """
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        text = ax.text(-0.2, 1.05, "Aribitrary text", transform=ax.transAxes)
        ax.set_title("Trigonometry")
        ax.grid('on')
        fig.savefig('samplefigure', bbox_extra_artists=(lgd, text), bbox_inches='tight')
        """


def generate_matrix(experiment_id, scale=1.):

    #experiment_id = "example_100_3"

    ######
    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    file_name = os.path.join(file_name, "experiments", str(experiment_id), "controllers", str(experiment_id) + ".txt")
    file_controllers = open(file_name, 'r')

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

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    file_name = os.path.join(file_name, "experiments", str(experiment_id), "results", "distances", str(experiment_id) + ".txt")
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

    for i in range(third_line):
        line = file_.readline().split(' ')
        a = code[int(line[0])]
        b = code[int(line[1])]
        value = float(line[2])
        matrix[a][b] += value
        quan[a][b] += 1

    for i in range(second_line):
        for j in range(i, second_line):
            matrix[i][j] /= float(quan[i][j])
            matrix[i][j] *= scale
            matrix[i][j] = int(round(matrix[i][j], 0))
            matrix[j][i] = matrix[i][j]

    file_.close()

    order = [i for i in range(num_families)]

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    file_name = os.path.join(file_name, "experiments", str(experiment_id), "controllers", str(experiment_id) + "_matrix.txt")
    file_ = open(file_name, 'r')

    num_families_order = int(file_.readline())

    for i in range(num_families_order):
        line = str(file_.readline().replace("\n","").replace(" ", ""))
        for j in range(num_families):
            if family_full_name[j].replace(" ","") == line:
                order[i] = j

    fig, ax = plt.subplots()

    matrix_order = [[0. for _ in range(num_families_order)] for _ in range(num_families_order)]

    for i in range(num_families_order):
        for j in range(num_families_order):
            c = int(matrix[order[i]][order[j]])
            matrix_order[i][j] = c
            ax.text(i, j, str(c), va='center', ha='center')

    family_full_name_order = []
    for i in range(num_families_order):
        family_full_name_order.append(family_full_name[order[i]])

    ax.matshow(matrix_order, cmap=plt.cm.Blues)

    x_values = family_full_name_order
    y_values = family_full_name_order
    y_axis = np.arange(0, num_families_order, 1)
    x_axis = np.arange(0, num_families_order, 1)

    plt.yticks(y_axis, y_values)
    plt.xticks(x_axis, x_values, rotation='vertical')

    file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..')
    file_name = os.path.join(file_name, "images", str(experiment_id) + "_matrix.png")
    plt.savefig(file_name)
    plt.show()



