import voting.modern as mo
import math
import voting.canonical as can
import os
import matplotlib.pyplot as plt

# MAP
def print_main_map():
    experiment_id = "testbed_100_100"
    mo.print_2d(experiment_id, mask=True, saveas="main_map", ms=16, tex=True)


def print_highest_borda():

    def normalizing_func(shade):
        shade -= 5000.
        shade /= 4900.
        shade = 1. - shade
        return shade

    def reversed_func(arg):
        arg = 1. - arg
        arg *= 4900
        arg += 5000
        return arg

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    print(reversed_func(0))

    #[print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    experiment_id = "testbed_100_100"
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['9900', '8675', '7450', '6225', '5000'])
    mo.print_2d(experiment_id, mask=True, saveas="highest_borda", ms=16, values="highest_borda",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_highest_plurality():

    def normalizing_func(shade):
        shade /= 100
        shade = 1 - shade
        return shade

    def reversed_func(arg):
        arg = 1 - arg
        arg *= 100
        return arg

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    print(normalizing_func(100))

    #[print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    experiment_id = "testbed_100_100"
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['100', '75', '50', '25', '0'])
    mo.print_2d(experiment_id, mask=True, saveas="highest_plurality", ms=16, values="highest_plurality",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_highest_copeland():

    def normalizing_func(shade):
        shade /= 99
        shade = 1 - shade
        shade *= 8
        return shade

    def reversed_func(arg):
        arg /= 8
        arg = 1 - arg
        arg *= 99
        return arg

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    experiment_id = "testbed_100_100"
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['99', '95.6', '92.8', '89.1', '86.6'])
    mo.print_2d(experiment_id, mask=True, saveas="highest_copeland", ms=16, values="highest_copeland",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_highest_dodgson():

    def normalizing_func(shade):
        if shade != 0:
            shade = math.log(shade) / 5.5
        return shade

    def reversed_func(arg):
        return math.e**(arg*5.5)

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    experiment_id = "testbed_100_100"
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '4', '16', '62', '245+'])
    mo.print_2d(experiment_id, mask=True, saveas="highest_dodgson", ms=16, values="highest_dodgson",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_greedy_removal():

    def marker_func(shade):
        if shade == 0.5:
            return 'x'
        return "o"

    def normalizing_func(shade):
        if shade > 0.05:
            shade = 0.05

        if shade < -0.05:
            shade = -0.05

        if shade != 0:
            shade = 0.5 + shade*10
        else:
            shade = 0.5

        return shade

    experiment_id = "testbed_100_100"
    custom_map = mo.custom_div_cmap(colors=["darkred", "red", "grey", "blue", "darkblue"])
    xticklabels = list(['1.05+', '1.025', '1', '1.025', '1.05+'])
    mo.print_2d(experiment_id, mask=True, saveas="greedy_removal", ms=16, values="greedy_removal",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                cmap=custom_map, marker_func=marker_func, tex=True, black=True)


def print_hb_time():

    def normalizing_func(shade):
        return shade

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    experiment_id = "testbed_100_100"
    #custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    custom_map = mo.custom_div_cmap(colors=["white", "purple"])
    xticklabels = list(['0s', '25s', '50s', '75s', '100s+'])
    mo.print_2d(experiment_id, mask=True, saveas="hb_time", ms=16, values="hb_time",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


# NOT MAP
def print_excel_mallows(experiment_id, target):

    for num_candidates in [5, 10, 20, 50, 100]:

        file_name = str(num_candidates) + ".txt"

        if target == "identity":
            file_name = "urn_model_id_" + file_name
        elif target == "uniformity":
            file_name = "urn_model_un_" + file_name

        path = os.path.join(os.getcwd(), "experiments", experiment_id, "results", "excel", file_name)
        file_ = open(path, 'r')

        keys = []
        values = []
        #for i in range(101):
        for i in range(51):
            line = file_.readline().strip().replace(" ", "").split(',')
            keys.append(float(line[0]))
            values.append(float(line[1]) / can.diagonal_math(num_candidates))
        file_.close()

        plt.plot(keys, values, label=str(num_candidates))

    plt.xlabel("Urn Model parameter")
    plt.ylabel("Normiazlied distance from " + str(target))
    plt.xlim([0, 5])
    plt.ylim([0, 1])
    plt.legend()

    path = os.path.join(os.getcwd(), "images", "urn_model_" + str(target))
    plt.savefig(path, bbox_inches='tight')
    plt.show()

