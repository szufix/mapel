import os


def make_folder_if_do_not_exist(path):
    is_exist = os.path.exists(path)
    if not is_exist:
        os.makedirs(path)

def is_module_loaded(module_import_name):
   '''
   Checks if a given module has already been loaded.
   BEWARE: the argument should be the true name of a module, not an alias. That
   is for
   > import foo as bar
   one should pass "foo" to this function
   '''
   import sys
   return module_import_name in sys.modules

def get_instance_id(single, family_id, j):
    if single:
        return family_id
    return f'{family_id}_{j}'


def rotate(vector, shift):
    shift = shift % len(vector)
    return vector[shift:] + vector[:shift]


def get_vector(type_id, num_candidates):
    if type_id == "uniform":
        return [1.] * num_candidates
    elif type_id == "linear":
        return [(num_candidates - x) for x in range(num_candidates)]
    elif type_id == "linear_low":
        return [(float(num_candidates) - float(x)) / float(num_candidates)
                for x in range(num_candidates)]
    elif type_id == "square":
        return [(float(num_candidates) - float(x)) ** 2 / float(num_candidates) ** 2 for x in
                range(num_candidates)]
    elif type_id == "square_low":
        return [(num_candidates - x) ** 2 for x in range(num_candidates)]
    elif type_id == "cube":
        return [(float(num_candidates) - float(x)) ** 3 / float(num_candidates) ** 3 for x in
                range(num_candidates)]
    elif type_id == "cube_low":
        return [(num_candidates - x) ** 3 for x in range(num_candidates)]
    elif type_id == "split_2":
        values = [1.] * num_candidates
        for i in range(num_candidates / 2):
            values[i] = 10.
        return values
    elif type_id == "split_4":
        size = num_candidates / 4
        values = [1.] * num_candidates
        for i in range(size):
            values[i] = 1000.
        for i in range(size, 2 * size):
            values[i] = 100.
        for i in range(2 * size, 3 * size):
            values[i] = 10.
        return values
    else:
        return type_id

# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 17.08.2022 #
# # # # # # # # # # # # # # # #
