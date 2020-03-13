from .voting import modern as mo


def test():

    print("Welcome to Mapel!")


def print_2d(name):

    mo.print_2d(name)


def print_matrix(name, scale=1.):

    mo.generate_matrix(name, scale)