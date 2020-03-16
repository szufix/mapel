Introduction
=============================
**mapel** (map of elections) is a python package that serves for drawing map of elections.


Installation
-----------------------------
::

    pip install mapel

Import
-----------------------------
::

    import mapel


Test
-----------------------------
::

    mapel.test()



Functionalities
=============================


Print the map of elections
-----------------------------
**print_2d** function is printing a two dimensional embedding of all the elections from a given experiment.
::

    mapel.print_2d(name, winners=0, angle=0, mask=False)

name
  : obligatory, name of the experiment.
  
winners
  : optional, number of winners of greedy CC to be marked.

angle
  : optional, rotate the image by *angle*.
    
mask
  : optional (only for *example_100_100*), mark all families on the map.


Print the matrix with distances
-----------------------------
**print_matrix** function is printing an array with average distances between each family of elections from a given experiment.

::

    mapel.print_matrix(name, scale=1.)


name
  : obligatory, name of the experiment.
  

scale
  : optional, multiply all the values by *scale*.

Experiments
=============================
Mixture of 800 election from 30 different  models: 

- 30x(each), Impartial Culture, Single Crossing, SPOC, Single Peaked (by Walsh), Single Peaked (by Conitzer),
- 30x(each) Euclidean: 1D Interval, 2D Square, 3D Cube, 5D Cube, 10D Cube 20D Cube, 2D Sphere, 3D Sphere, 5D Sphere,  
- 30x(each) Urn Model with the following parameter: 0.5, 0.2, 0.1, 0.05, 0.02, 0.01 
- 20x(each) Mallows with the following parameter: 0.999, 0.99, 0.95, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001

List of examples
-----------------------------
- Example 1: 100 voters, 100 candidates; name: **example_100_100**
- Example 2: 100 voters, 20 candidates; name: **example_100_20**
- Example 3: 100 voters, 10 candidates; name: **example_100_10**
- Example 4: 100 voters, 5 candidates; name: **example_100_5**
- Example 5: 100 voters, 4 candidates; name: **example_100_4**
- Example 6: 100 voters, 3 candidates; name: **example_100_3**

Example of use::

    mapel.print_2d("example_100_20", winners=50)
    
Experiment structure: tbu
    
Extras
=============================

Matrix with distances
-----------------------------
In *experiments/name_of_the_experiment/controllers/name_of_the_experiment_matrix.txt* there a is list of names of the families of elections. The number of families and their order can be change and will influence the *mapel.print_matrix()* function.

SOC files
-----------------------------
In *experiments/name_of_the_experiment/elections/soc_approx_cc/* are all the elections ordered by greedy CC.

Definition of the soc format can be found here: http://www.preflib.org/data/format.php#soc

