"""
Script which generates map.csv file for experiment with syntetic elections of varying number of candidates.
The statictical culture is always displayed in the same color with varying levels of transparency depending on the number of candidates - by default lighter shade for lowe number of candidates
This function can be run
Optional parameters (bools by default true unless stated otherwise):
    candidates: set to [8, 9, 10, 11, 12] by default, meaning
    alphas: transparency for the elections with number of candidates on corresponding indeces in the candidates list
    family_size: number of elections per family
    num_voters: number of voters per election
    file_name: file to which the output should be saved
    incl_ic: bool whether IC culture should be included
    incl_iac: bool whether IAC culture should be included
    incl_conitzer: bool whether conitzer culture should be included
    incl_walsh: bool whether walsh culture should be included
    incl_sc: bool whether single crossing culture should be included
    incl_mallows_default: bool whether Mallows with default values of phi (0.3, 0.5, 0.7) should be included
    incl_mallows_custom: bool whether Mallows with custom values of phi should be included - set by default to False, if set to True, then list of phi also needs to be added
    mallows_params_phi: custom phis that will be added into the csv file
    mallows_params_colors: colors of corresponding custom phis for Mallows elections
    incl_euclidean: bool whether euclidean cultures should be included
    incl_extremes: bool whether UN, AN, ID and IDAN should be included

"""

def generate_mapcsv_contents(*args, **kwargs):
    candidates = [8, 9, 10, 11, 12]
    alphas = [0.2, 0.4, 0.6, 0.8, 1]
    family_size = 10
    urn_family_size = 10
    mallows_family_size = 10
    num_voters = 100
    file_name = "map.txt"
    incl_ic = True
    incl_iac = False
    incl_urn = True
    incl_conitzer = True
    incl_walsh = True
    incl_sc = True
    incl_spoc = True
    incl_mallows_default = True
    incl_mallows_custom = False
    mallows_params_phi = None
    mallows_params_colors = None
    incl_euclidean = True
    incl_extremes = True

    for key, val in kwargs.items():
        if key == "candidates":
            candidates = val
        elif key == "alphas":
            alphas = val
        elif key == "family_size":
            family_size = val
        elif key == "num_voters":
            num_voters = val
        elif key == "file_name":
            file_name = val
        elif key == "incl_ic":
            incl_ic = val
        elif key == "incl_iac":
            incl_iac = val
        elif key == "incl_conitzer":
            incl_conitzer = val
        elif key == "incl_walsh":
            incl_walsh = val
        elif key == "incl_sc":
            incl_sc = val
        elif key == "incl_spoc":
            incl_spoc = val
        elif key == "incl_mallows_default":
            incl_mallows_default = val
        elif key == "incl_mallows_custom":
            incl_mallows_default = False
            incl_mallows_custom = True
        elif key == "mallows_params_phi":
            mallows_params_phi = val
        elif key == "mallows_params_colors":
            mallows_params_colors = val
        elif key == "incl_euclidean":
            incl_euclidean = val
        elif key == "incl_extremes":
            incl_extremes = val
        else:
            print("Optional argument \"", key, "\" does not exist.")

    if incl_mallows_custom is True and (mallows_params_phi is None or mallows_params_colors is None):
        incl_mallows_custom = False
        print("Missing parameter values for custom mallows, so no custom mallows will be added.")

    file = open(file_name, 'w')
    file.write("size;num_candidates;num_voters;culture_id;params;color;alpha;family_id;marker;path;label\n")

    for i in range(0, len(candidates)):
        if incl_ic:
            ic_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "ic" + ";" + "{}" + ";" + "black" + ";" + str(
                alphas[i]) + ";" + "ic" + str(candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "IC \n"
            file.write(ic_string)

        if incl_iac:
            iac_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "iac" + ";" + "{}" + ";" + "lavender" + ";" + str(
                alphas[i]) + ";" + "iac" + str(candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "IAC \n"
            file.write(iac_string)

        if incl_urn:
            urn_alpha = [0.01, 0.02, 0.05, 0.1, 0.5]
            urn_alpha_color = ['palegoldenrod', 'yellow', 'gold', 'orange', 'red']
            for j in range(0, len(urn_alpha)):
                urn_string = str(urn_family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "urn" + ";" + "{'alpha': " + str(urn_alpha[j]) + "}" + ";" + str(urn_alpha_color[j]) + ";" + str(
                alphas[i]) + ";" + "urn" + str(candidates[i]) + '-' + str(urn_alpha[j]) + ";" + "o" + ";" + "{}" + ";" + "Urn-" + str(urn_alpha[j]) + "\n"
                file.write(urn_string)

        if incl_conitzer:
            conitzer_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "conitzer" + ";" + "{}" + ";" + "saddlebrown" + ";" + str(
                alphas[i]) + ";" + "conitzer" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "SP-Conitzer \n"
            file.write(conitzer_string)

        if incl_walsh:
            walsh_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "walsh" + ";" + "{}" + ";" + "olive" + ";" + str(
                alphas[i]) + ";" + "walsh" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "SP-Walsh \n"
            file.write(walsh_string)

        if incl_sc:
            singlecrossing_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "single-crossing" + ";" + "{}" + ";" + "mediumorchid" + ";" + str(
                alphas[i]) + ";" + "single-crossing" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "SC \n"
            file.write(singlecrossing_string)

        if incl_mallows_default:
            phis = [0.001, 0.01, 0.1, 0.5, 0.75, 0.99]
            phis_colors = ['paleturquoise', 'skyblue', 'dodgerblue', 'cornflowerblue', 'blue', 'navy']
            for j in range(0, len(phis)):
                mallows_string = str(mallows_family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "norm-mallows" + ";" + "{'normphi':" + str(phis[j]) + "}" + ";" + phis_colors[j] + ";" + str(
                alphas[i]) + ";" + "norm-mallows-" + str(candidates[i]) + '-' + str(phis[j]) + ";" + "o" + ";" + "{}" + ";" + "Norm-Mallows-" + str(phis[j]) + "\n"
                file.write(mallows_string)

        if incl_mallows_custom:
            for j in range(0, len(mallows_params_phi)):
                mallows_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                    num_voters) + ";" + "norm-mallows" + ";" + "{'normphi': " + str(mallows_params_phi[j]) +"}" + ";" + mallows_params_colors[j] + ";" + str(
                    alphas[i]) + ";" + "norm-mallows" + str(candidates[i]) + str(
                    mallows_params_phi[j]) + ";" + "x" + ";" + "{}" + ";" + "Norm-Mallows-" +str(mallows_params_phi[j]) +"\n"
                file.write(mallows_string)

        if incl_spoc:
            spoc_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "spoc_conitzer" + ";" + "{}" + ";" + "firebrick" + ";" + str(
                alphas[i]) + ";" + "spoc_conitzer" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "SPOC \n"
            file.write(spoc_string)


        if incl_euclidean:
            euclidean_1_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "euclidean" + ";" + "{'dim': 1, 'space': 'uniform'}" + ";" + "lime" + ";" + str(
                alphas[i]) + ";" + "Interval" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "1D Interval \n"
            euclidean_2_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "euclidean" + ";" + "{'dim': 2, 'space': 'uniform'}" + ";" + "green" + ";" + str(
                alphas[i]) + ";" + "Square" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "Square \n"
            euclidean_3_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "euclidean" + ";" + "{'dim': 3, 'space': 'uniform'}" + ";" + "forestgreen" + ";" + str(
                alphas[i]) + ";" + "Cube" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "3-Cube \n"
            euclidean_5_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "euclidean" + ";" + "{'dim': 5, 'space': 'uniform'}" + ";" + "palegreen" + ";" + str(
                alphas[i]) + ";" + "5-Cube" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "5-Cube \n"
            euclidean_10_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "euclidean" + ";" + "{'dim': 10, 'space': 'uniform'}" + ";" + "yellowgreen" + ";" + str(
                alphas[i]) + ";" + "10-Cube" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "10-Cube \n"
            euclidean_2_sphere_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "euclidean" + ";" + "{'dim': 2, 'space': 'sphere'}" + ";" + "deeppink" + ";" + str(
                alphas[i]) + ";" + "Circle" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "Circle \n"
            euclidean_3_sphere_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "euclidean" + ";" + "{'dim': 3, 'space': 'sphere'}" + ";" + "lightpink" + ";" + str(
                alphas[i]) + ";" + "Sphere" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "Sphere \n"
            file.write(euclidean_1_string)
            file.write(euclidean_2_string)
            file.write(euclidean_3_string)
            file.write(euclidean_5_string)
            file.write(euclidean_10_string)
            file.write(euclidean_2_sphere_string)
            file.write(euclidean_3_sphere_string)

    if incl_extremes:
        st_string = '1' + ';' + str(15) + ';' + '50' + ';real_stratification;{};black;' + str(1) + ';ST;x;{};ST \n'
        un_string = '1' + ';' + str(15) + ';' + '50' + ';un_from_matrix;{};black;' + str(1) + ';UN;x;{};UN \n'
        id_string = '1' + ';' + str(15) + ';' + '50' + ';real_identity;{};rosybrown;' + str(1) + ';ID;x;{};ID \n'
        an_string = '1' + ';' + str(15) + ';' + '50' + ';real_antagonism;{};royalblue;' + str(1) + ';AN;x;{};AN \n'
        idan_string = '4' + ';' + str(15) + ';' + '50' + ';idan_part;{};midnightblue;' + str(1) + ';IDAN;x;{\'variable\' : \'part_share\'};IDAN \n'
        file.write(st_string)
        file.write(un_string)
        file.write(id_string)
        file.write(an_string)
        file.write(idan_string)

    file.close()


if __name__ == "__main__":
    generate_mapcsv_contents()
