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
If everthing was correctly downloaded and imported then after writing the following command:
::

    mapel.test()

the program should print "Welcome to Mapel!".


Functionalities
=============================


Print the map of elections
-----------------------------
**print_2d** function is printing a two dimensional embedding of all the elections from a given experiment.
::

    mapel.print_2d(exp_name, num_elections=800, main_order="", num_winners=0,  winners_order="approx_cc", values="default", coloring="purple", mask=False, angle=0) 

exp_name
  : obligatory, name of the experiment.
  
num_winners
  : optional, number of winners of greedy CC to be marked.
  
winners_order
  : optional, order in which the winners should appear.
  
num_elections
  : optional, number of elections/points to be printed.
  
main_order
  : optional, order in which the elections/points should appear.
  
values
  : optional, ...
  
coloring
  : optional, ...
  
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

List of experiments
-----------------------------
- Experiment 1: 100 voters, 100 candidates; exp_name: **example_100_100**
- Experiment 2: 100 voters, 20 candidates; exp_name: **example_100_20**
- Experiment 3: 100 voters, 10 candidates; exp_name: **example_100_10**
- Experiment 4: 100 voters, 5 candidates; exp_name: **example_100_5**
- Experiment 5: 100 voters, 4 candidates; exp_name: **example_100_4**
- Experiment 6: 100 voters, 3 candidates; exp_name: **example_100_3**

Example of use::

    mapel.print_2d("example_100_20", winners=50)
    
Experiment structure: tbu


Examples
=============================

Your own experiment
-----------------------------

    
Extras
=============================

Matrix with distances
-----------------------------
In *experiments/name_of_the_experiment/controllers/name_of_the_experiment_matrix.txt* there a is list of names of the families of elections. The number of families and their order can be change and will influence the *mapel.print_matrix()* function.

SOC files
-----------------------------
In *experiments/name_of_the_experiment/elections/soc_approx_cc/* are all the elections ordered by greedy CC.

Definition of the soc format can be found here: http://www.preflib.org/data/format.php#soc

