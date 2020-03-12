Installation
=============================
::

    pip install mapel

Import
=============================
::

    import mapel


Test
=============================
::

    mapel.test()

Experiments
=============================
Mixture of 800 election from 30 different  models: 

- 30x(each), Impartial Culture, Single Crossing, SPOC, Single Peaked (by Walsh), Single Peaked (by Conitzer),
- 30x(each) Euclidean: 1D Interval, 2D Square, 3D Cube, 5D Cube, 10D Cube 20D Cube, 2D Sphere, 3D Sphere, 5D Sphere,  
- 30x(each) Urn Model with the following parameter: 0.5, 0.2, 0.1, 0.05, 0.02, 0.01 
- 20x(each) Mallows with the following parameter: 0.999, 0.99, 0.95, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001

List of examples
-----------------------------
- Example 1: 100 voters, 100 candidates, example_100_100
- Example 2: 100 voters, 20 candidates, example_100_20
- Example 3: 100 voters, 10 candidates, example_100_10
- Example 4: 100 voters, 5 candidates, example_100_5
- Example 5: 100 voters, 4 candidates, example_100_4
- Example 6: 100 voters, 3 candidates, example_100_3

Functionalities
=============================


Print the map of elections
-----------------------------
::

    mapel.print_2d(name="...")


Print the matrix with distances
-----------------------------
::

    mapel.print_matrix(name, scale=1.)

Obligaotry paramteres: name

Optional parameters: scale (by deafalut =1.)
