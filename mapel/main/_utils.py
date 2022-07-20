
def get_instance_id(single, family_id, j):
    if single:
        return family_id
    return f'{family_id}_{j}'


def rotate(vector, shift):
    shift = shift % len(vector)
    return vector[shift:] + vector[:shift]


def get_vector(type, num_candidates):
    if type == "uniform":
        return [1.] * num_candidates
    elif type == "linear":
        return [(num_candidates - x) for x in range(num_candidates)]
    elif type == "linear_low":
        return [(float(num_candidates) - float(x)) / float(num_candidates) for x in range(num_candidates)]
    elif type == "square":
        return [(float(num_candidates) - float(x)) ** 2 / float(num_candidates) ** 2 for x in
                range(num_candidates)]
    elif type == "square_low":
        return [(num_candidates - x) ** 2 for x in range(num_candidates)]
    elif type == "cube":
        return [(float(num_candidates) - float(x)) ** 3 / float(num_candidates) ** 3 for x in
                range(num_candidates)]
    elif type == "cube_low":
        return [(num_candidates - x) ** 3 for x in range(num_candidates)]
    elif type == "split_2":
        values = [1.] * num_candidates
        for i in range(num_candidates / 2):
            values[i] = 10.
        return values
    elif type == "split_4":
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
        return type
