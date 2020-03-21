Introduction
=============================
**mapel** (map of elections) is a python package that serves for drawing map of elections. It is a testbed of elections to be used
for various election-related experiments such as testing algorithms
or estimating the frequency of a given phenomenon.

For more details please look at:

    Szufa,  S.,  Faliszewski,  P.,  Skowron,  P.,  Slinko,  A.,  Talmon,  N.:  Drawing  a  map of elections in the space of statistical cultures. In: Proceedings of AAMAS-2020, to appear.


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
  : obligatory, string; name of the experiment.
  
num_winners
  : optional, int; number of winners of greedy CC to be marked.
  
winners_order
  : optional, string, name of the file that contains the order in which the winners should appear.
  
num_elections
  : optional, int, number of points to be printed.
  
main_order
  : optional, string; name of the file that contains the order in which the points should appear.
  
values
  : optional, string; name of the file that contains alpha values.
  
coloring
  : optional, string; color in which all the points should appear. If set to "intervals" then it will color all points from [0.8,1] red, [0.6,0.8) orange, [0.4,0.6) yellow, [0.2,0.4) green, [0,0.2) blue.
  
angle
  : optional, float; rotate the image by *angle*.
    
mask
  : optional, bool; mark all families on the map (only for *example_100_100*).


Print the matrix with distances
-----------------------------
**print_matrix** function is printing an array with average distances between each family of elections from a given experiment.

::

    mapel.print_matrix(exp_name, scale=1.)


exp_name
  : obligatory, string; name of the experiment.
  

scale
  : optional, string; multiply all the values by *scale*.


Prepare SOC files
-----------------------------
**prepare_approx_cc_order** [the name of this function should be changed]
::

    mapel.print_matrix(exp_name)


exp_name
  : obligatory, name of the experiment.
  
  
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
    
Experiment structure (after downloading mapel): 

::

    exp_name
    ├── controllers.py     
    │   ├── map.py
    │   └── matrix.py
    ├── elections          
    │   ├── soc_approx_cc 
    │   │   ├── (empty)
    │   └── soc_original
    │       └── (800 txt files with elections)
    └── results
        ├── distances
        │   ├── positionwise.txt
        │   └── positionwise_info.txt
        ├── points
        │   └── 2d.txt
        └── winners
            └── approx_cc.txt


Examples
=============================

Simple example of use

    mapel.print_2d("example_100_20", winners=50)


Your own experiment
-----------------------------



    
Extras
=============================

Matrix with distances
-----------------------------
If you want to print just several selected families of elections or change the order in which they appear you should go to the file:  "*experiments/exp_name/controllers/exp_name_matrix.txt*". There a is list of names of all the families of elections. The number of families and their order can be change and will influence the *mapel.print_matrix()* function.

SOC files
-----------------------------
Definition of the soc format can be found here: http://www.preflib.org/data/format.php#soc



Contact
=============================
If you have any questions or have found a bug please email me at *stanislaw.szufa@uj.edu.pl*
