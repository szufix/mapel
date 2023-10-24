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

"""

def generate_mapcsv_contents(*args, **kwargs):
    candidates = [8, 9, 10, 11, 12]
    alphas = [0.2, 0.4, 0.6, 0.8, 1]
    family_size = 10
    num_voters = 100
    file_name = "map_file.txt"
    incl_ic = True
    incl_iac = True
    incl_conitzer = True
    incl_walsh = True
    incl_sc = True
    incl_mallows_default = True
    incl_mallows_custom = False
    mallows_params_phi = None
    mallows_params_colors = None

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
        elif key == "incl_mallows_default":
            incl_mallows_default = val
        elif key == "incl_mallows_custom":
            incl_mallows_default = False
            incl_mallows_custom = True
        elif key == "mallows_params_phi":
            mallows_params_phi = val
        elif key == "mallows_params_colors":
            mallows_params_colors = val
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
                num_voters) + ";" + "ic" + ";" + "{}" + ";" + "green" + ";" + str(
                alphas[i]) + ";" + "ic" + str(candidates[i]) + ";" + "x" + ";" + "{}" + ";" + "IC-candidates:" + str(
                candidates[i]) + "\n"
            file.write(ic_string)
        if incl_iac:
            iac_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "iac" + ";" + "{}" + ";" + "yellow" + ";" + str(
                alphas[i]) + ";" + "iac" + str(candidates[i]) + ";" + "x" + ";" + "{}" + ";" + "IAC-candidates:" + str(
                candidates[i]) + "\n"
            file.write(iac_string)
        if incl_conitzer:
            conitzer_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "conitzer" + ";" + "{}" + ";" + "red" + ";" + str(
                alphas[i]) + ";" + "conitzer" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "Conitzer-candidates:" + str(candidates[i]) + "\n"
            file.write(conitzer_string)
        if incl_walsh:
            walsh_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "walsh" + ";" + "{}" + ";" + "orange" + ";" + str(
                alphas[i]) + ";" + "walsh" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "Walsh-candidates:" + str(
                candidates[i]) + "\n"
            file.write(walsh_string)
        if incl_sc:
            singlecrossing_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "single-crossing" + ";" + "{}" + ";" + "maroon" + ";" + str(
                alphas[i]) + ";" + "single-crossing" + str(
                candidates[i]) + ";" + "o" + ";" + "{}" + ";" + "SC-candidates:" + str(candidates[i]) + "\n"
            file.write(singlecrossing_string)
        if incl_mallows_default:
            normmallows05_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "norm-mallows" + ";" + "{'normphi': 0.5}" + ";" + "blue" + ";" + str(
                alphas[i]) + ";" + "norm-mallows" + str(candidates[i]) + str(
                0.5) + ";" + "x" + ";" + "{}" + ";" + "Norm-Mallows-0.5-candidates:" + str(candidates[i]) + "\n"
            normmallows03_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "norm-mallows" + ";" + "{'normphi': 0.3}" + ";" + "purple" + ";" + str(
                alphas[i]) + ";" + "norm-mallows" + str(candidates[i]) + str(
                0.3) + ";" + "x" + ";" + "{}" + ";" + "Norm-Mallows-0.3-candidates:" + str(candidates[i]) + "\n"
            normmallows07_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                num_voters) + ";" + "norm-mallows" + ";" + "{'normphi': 0.7}" + ";" + "black" + ";" + str(
                alphas[i]) + ";" + "norm-mallows" + str(candidates[i]) + str(
                0.7) + ";" + "x" + ";" + "{}" + ";" + "Norm-Mallows-0.7-candidates:" + str(candidates[i]) + "\n"
            file.write(normmallows03_string)
            file.write(normmallows05_string)
            file.write(normmallows07_string)
        if incl_mallows_custom:
            for j in range(0, len(mallows_params_phi)):
                mallows_string = str(family_size) + ";" + str(candidates[i]) + ";" + str(
                    num_voters) + ";" + "norm-mallows" + ";" + "{'normphi': " + str(mallows_params_phi[j]) +"}" + ";" + mallows_params_colors[j] + ";" + str(
                    alphas[i]) + ";" + "norm-mallows" + str(candidates[i]) + str(
                    mallows_params_phi[j]) + ";" + "x" + ";" + "{}" + ";" + "Norm-Mallows-" + str(mallows_params_phi[j]) + "-candidates:" + str(candidates[i]) + "\n"
                file.write(mallows_string)
    file.close()


if __name__ == "__main__":
    generate_mapcsv_contents()
