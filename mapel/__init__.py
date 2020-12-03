import voting.print as mo
import math
import voting.canonical as can
import os
import matplotlib.pyplot as plt
import voting.metrics as metr



def hello():
    print("Hello!")

# CORRELATION
def print_corr_highest_borda(experiment_id):
    values = "highest_borda"
    mo.print_param_vs_distance(experiment_id, values=values, scale="none", saveas=values)


def print_corr_highest_plurality(experiment_id):
    values = "highest_plurality"
    mo.print_param_vs_distance(experiment_id, values=values, scale="none", saveas=values)


def print_corr_highest_copeland(experiment_id):
    values = "highest_copeland"
    mo.print_param_vs_distance(experiment_id, values=values, scale="none", saveas=values)


def print_corr_highest_dodgson(experiment_id):
    values = "highest_dodgson"
    mo.print_param_vs_distance(experiment_id, values=values, scale="none", saveas=values)


def print_corr_highest_dodgson_time(experiment_id):
    values = "highest_dodgson_time"
    mo.print_param_vs_distance(experiment_id, values=values, scale="none", saveas=values)


def print_corr_borda_time(experiment_id):
    values = "hb_time"
    mo.print_param_vs_distance(experiment_id, values=values, scale="log", saveas=values)



# MAP
def print_main_map():
    experiment_id = "testbed_100_100"
    mo.print_2d(experiment_id, mask=True, saveas="main_map", ms=16, tex=True)


def print_highest_borda(experiment_id):

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

    #[print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    #custom_map = mo.custom_div_cmap(colors=["yellow", "orange", "red", "purple", "blue"])

    xticklabels = list(['9900', '8675', '7450', '6225', '5000'])
    mo.print_2d(experiment_id, mask=True, saveas="highest_borda", ms=16, values="highest_borda",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_zip():

    def normalizing_func(shade):

        return shade

    def reversed_func(arg):

        return arg

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    print(reversed_func(0))

    #[print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    experiment_id = "testbed_100_100"
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    #custom_map = mo.custom_div_cmap(colors=["yellow", "orange", "red", "purple", "blue"])

    xticklabels = list(['0', '0.25', '0.5', '0.75', '1'])
    mo.print_2d(experiment_id, mask=True, saveas="zip_size", ms=16, values="zip_size",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_highest_plurality(experiment_id):

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


    #[print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['100', '75', '50', '25', '0'])
    mo.print_2d(experiment_id, mask=True, saveas="highest_plurality", ms=16, values="highest_plurality",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_highest_copeland(experiment_id):

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

    #[print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    #xticklabels = list(['99', '95.6', '92.8', '89.1', '86.6'])
    xticklabels = list(['99', '96', '93', '90', '87'])
    mo.print_2d(experiment_id, mask=True, saveas="highest_copeland", ms=16, values="highest_copeland",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_highest_dodgson(experiment_id):

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

    #[print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

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
        shade -= 2
        shade = math.log(shade)
        shade /= 4.58
       # shade *= 100
        #shade = math.log(shade)
       # shade /= 5
        return shade


    def reversed_func(arg):
        arg *= 4.58
        arg = math.e**arg
        arg += 2
        return arg

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    experiment_id = "testbed_100_100"
    #custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    custom_map = mo.custom_div_cmap(colors=["white", "purple"])
    custom_map = mo.custom_div_cmap(colors=["bisque", "orange", "red", "purple", "black"])
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    #custom_map = mo.custom_div_cmap(colors=["#ffe4ff", "purple"])
    xticklabels = list(['2s', '5s', '12s', '33s', '100s+'])
    mo.print_2d(experiment_id, mask=True, saveas="hb_time", ms=16, values="hb_time",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, tex=True, black=True)


def print_highest_dodgson_time():
    # log scale.

    def normalizing_func(shade):
        shade /= 600
        return shade

    def reversed_func(arg):
        return arg*600

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    experiment_id = "testbed_100_100"
    #custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    custom_map = mo.custom_div_cmap(colors=["bisque", "orange", "red", "purple", "black"])
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    #custom_map = mo.custom_div_cmap(colors=["#ffe4ff", "purple"])
    xticklabels = list(['0', '150', '300', '400', '600+'])
    mo.print_2d(experiment_id, mask=True, saveas="highest_dodgson_time", ms=16, values="highest_dodgson_time",
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

################
### 100 x 10 ###
################


# MAPS - SCORES
def print_highest_plurality_10(experiment_id):

    def normalizing_func(shade):
        shade -= 10
        shade /= 90
        shade = 1 - shade
        return shade

    def reversed_func(arg):
        arg = 1 - arg
        arg *= 90
        arg += 10
        return arg

    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['100', '77.5', '55', '32.5', '10'])
    mo.print_2d(experiment_id, saveas="scores/map_highest_plurality_10", ms=16, values="highest_plurality",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                cmap=custom_map, black=True, levels=True)


def print_highest_borda_10(experiment_id):

    def normalizing_func(shade):
        if shade == -1:
            return 1
        shade -= 500.
        shade /= 400.
        shade = 1. - shade
        return shade

    def reversed_func(arg):
        arg = 1. - arg
        arg *= 400
        arg += 500
        return arg

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['900', '800', '700', '600', '500'])
    mo.print_2d(experiment_id, saveas="scores/map_highest_borda_10", ms=16, values="highest_borda",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                cmap=custom_map, black=True, levels=True)


def print_highest_borda12_10(experiment_id):

    def normalizing_func(shade):
        #shade -= 500.
        #shade /= 400.
        shade /= 900
        shade = 1. - shade
        return shade

    def reversed_func(arg):
        arg *= 900
        return arg

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    #custom_map = mo.custom_div_cmap(colors=["yellow", "orange", "red", "purple", "blue"])

    xticklabels = list(['900', '800', '700', '600', '500'])
    mo.print_2d(experiment_id, saveas="scores/map_highest_borda12_10", ms=16, values="highest_borda12",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                cmap=custom_map, black=True, levels=True)


def print_highest_copeland_10(experiment_id):

    def normalizing_func(shade):
        shade /= 9
        shade = 1 - shade
        shade *= 2
        return shade

    def reversed_func(arg):
        arg /= 2
        arg = 1 - arg
        arg *= 9
        return arg

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    #xticklabels = list(['99', '95.6', '92.8', '89.1', '86.6'])
    xticklabels = list(['9', '8', '7', '6', '5-'])
    mo.print_2d(experiment_id, saveas="scores/map_highest_copeland_10", ms=16, values="highest_copeland",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, black=True, levels=True)


def print_highest_dodgson_10(experiment_id):

    def normalizing_func(shade):
        if shade != 0:
            shade = math.log(shade) / 2
        return shade

    def reversed_func(arg):
        return math.e**(arg*2)

    def marker_func(shade):
        if shade == 0:
            return 'x'
        return "o"

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '1', '3', '5', '8+'])
    mo.print_2d(experiment_id, saveas="scores/map_highest_dodgson_10", ms=16, values="highest_dodgson",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                marker_func=marker_func, cmap=custom_map, black=True, levels=True)


# MAPS - DISTANCES

def print_distance_from_diameter_path_10(experiment_id):

    def normalizing_func(shade):
        #shade /= metr.map_diameter(10)
        #shade -= 1
        #shade *= 10
        shade /= 10
        return shade

    def reversed_func(arg):
        #arg /= 10
        #arg += 1
        #arg *= metr.map_diameter(10)
        arg *= 10
        return arg

    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '2.5', '5', '7.5', '10+'])
    mo.print_2d(experiment_id, saveas="distances/map_distance_from_diameter_10", ms=16, values="diameter",
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)


def print_distance_from_antagonism_10(experiment_id):

    def normalizing_func(shade):
        shade /= 25
        return shade

    def reversed_func(arg):
        arg *= 25
        return arg

    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '6.25', '12.5', '18.75', '25+'])
    mo.print_2d(experiment_id, saveas="distances/map_distance_from_antagonism_10", ms=16, values="antagonism",
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)


def print_distance_from_chess_10(experiment_id):

    def normalizing_func(shade):
        shade /= 25
        return shade

    def reversed_func(arg):
        arg *= 25
        return arg

    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '6.25', '12.5', '18.75', '25+'])
    mo.print_2d(experiment_id, saveas="distances/map_distance_from_chess_10", ms=16, values="chess",
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)


def print_distance_from_rand_point_10(experiment_id, rand_id=-1, nice_name=''):

    def normalizing_func(shade):
        shade /= 25
        return shade

    def reversed_func(arg):
        arg *= 25
        return arg

    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '6.25', '12.5', '18.75', '25+'])
    mo.print_2d(experiment_id, saveas="distances/map_rand_point_" + str(nice_name), ms=16, values="rand_point_" + str(nice_name),
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)


# MAPS - OTHERS
def print_distance_from_cztery_sery_10(experiment_id):

    def normalizing_func(shade):
        shade -= 64
        if shade < 0:
            print("LLLOOOOWWWW")
        shade /= 10
        return shade


    def reversed_func(arg):
        arg *= 10
        arg += 64
        return arg


    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['64', '66.5', '69', '71.5', '74+'])
    mo.print_2d(experiment_id, saveas="scores/map_cztery_sery_10", ms=16, values="cztery_sery",
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)


def print_distance_from_dwa_sery_10(experiment_id):

    def normalizing_func(shade):
        shade -= 27
        if shade < 0:
            print("LLLOOOOWWWW")
        shade /= 12
        return shade

    def reversed_func(arg):
        arg *= 12
        arg += 27
        return arg

    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['27', '30', '33', '36', '39+'])
    mo.print_2d(experiment_id, saveas="scores/map_dwa_sery_10", ms=16, values="dwa_sery",
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)


def print_crazy(experiment_id):

    def normalizing_func(shade):
        shade -= 1
        #shade = 1. - shade
        return shade

    def reversed_func(arg):
        arg *= 900
        return arg

    [print(reversed_func(i)) for i in [0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    #custom_map = mo.custom_div_cmap(colors=["yellow", "orange", "red", "purple", "blue"])

    xticklabels = list(['900', '800', '700', '600', '500'])
    mo.print_2d(experiment_id, saveas="scores/map_crazy", ms=16, values="crazy",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                cmap=custom_map, black=True, levels=True)


def print_smooth_levels_c10(experiment_id):

    def normalizing_func(shade):
        shade /= metr.map_diameter(10)
        return shade

    def reversed_func(arg):
        arg *= metr.map_diameter(10)
        return arg


    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '8.25', '16.5', '24.75', '33'])
    mo.print_2d(experiment_id, saveas="scores/map_smooth_levels_10", ms=16, values="levels",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                cmap=custom_map, black=True, levels=True)


def print_sharp_levels_c10(experiment_id, num_levels=5):

    def normalizing_func(shade):
        shade /= metr.map_diameter(10)
        shade *= num_levels
        shade = math.floor(shade)
        shade /= num_levels
        return shade

    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '8.25', '16.5', '24.75', '33'])
    mo.print_2d(experiment_id, saveas="scores/map_sharp_levels_10_l" + str(num_levels), ms=16, values="levels",
                normalizing_func=normalizing_func, xticklabels=xticklabels,
                cmap=custom_map, black=True, levels=True)


# CORRELATIONS
def print_correlation_10(experiment_id, values=None, ylabel_text='', target=''):
    mo.print_param_vs_distance(experiment_id, values=values, scale="none", target=target,
                               saveas=('corr_'+values+'_10'), ylabel_text=ylabel_text)


# ILP 100x6
def print_distance_from_diameter_ilp_6(experiment_id):
    def normalizing_func(shade):
        #shade /= metr.map_diameter_unid(6)
        # shade -= 1
        # shade *= 10
        shade /= 1.5
        return shade

    def reversed_func(arg):
        # arg /= 10
        # arg += 1
        # arg *= metr.map_diameter(10)
        arg *= 1.5
        return arg

    #print(metr.map_diameter_unid(6))

    [print(reversed_func(i)) for i in [0., 0.25, 0.5, 0.75, 1.]]
    custom_map = mo.custom_div_cmap(colors=["lightgreen", "yellow", "orange", "red", "black"])
    xticklabels = list(['0', '0.375', '0.75', '1.125', '1.5+'])
    mo.print_2d(experiment_id, saveas="distances/map_distance_from_diameter_ilp_6", ms=16, values="diameter_ilp",
                normalizing_func=normalizing_func, xticklabels=xticklabels, cmap=custom_map, black=True, levels=True)


# NEW 29.11.2020
def print_2d(experiment_id, **kwargs):
    mo.print_2d(experiment_id, **kwargs)

def print_matrix(experiment_id):
    mo.print_matrix(experiment_id)

def print_param_vs_distance(experiment_id, **kwargs):
    mo.print_param_vs_distance(experiment_id, **kwargs)

def prepare_approx_cc_order(experiment_id, **kwargs):
    mo.prepare_approx_cc_order(experiment_id, **kwargs)