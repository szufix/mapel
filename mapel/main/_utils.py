
def get_instance_id(single, family_id, j):
    if single:
        return family_id
    return f'{family_id}_{j}'


def rotate(vector, shift):
    shift = shift % len(vector)
    return vector[shift:] + vector[:shift]