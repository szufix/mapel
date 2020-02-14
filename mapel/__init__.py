from . import modern as mo

def hello():
    print("hello")

def test():

    #name = str(sys.argv[1])
    name = "example_100_20"

    #mo.convert_xd_to_2d(name, num_iterations=1000)

    mo.print_2d(name, shades=False)
