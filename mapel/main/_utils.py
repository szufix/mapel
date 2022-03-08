
def get_instance_id(single, family_id, j):
    if single:
        return family_id
    return f'{family_id}_{j}'

