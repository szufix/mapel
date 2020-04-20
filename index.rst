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
  : optional, string; name of the file that contains alpha values. The file should be in *exp_name/controler/* folder.
  
coloring
  : optional, string; color in which all the points should appear (use this only if *values* is not equall to *default*). If set to "intervals" then it will color all points from [0.8,1] red, [0.6,0.8) orange, [0.4,0.6) yellow, [0.2,0.4) green, [0,0.2) blue.
  
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
**prepare_approx_cc_order** funtion serves for preparing elections in soc format in approx_cc order. This function is just coping files from *soc_original* and pasting them in an order from winners *exp_name/results/winners/appro_cc.txt*. 

::

    mapel.prepare_approx_cc_order(exp_name)


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
    ├── controllers     
    │   ├── basic
    │   │   ├── map.txt
    │   │   └── matrix.txt
    │   └── advance
    │       ├── hb_time.txt (only in example_100_100)
    │       ├── positionwise_approx_cc_greedy.txt (only in example_100_100)           
    │       ├── positionwise_approx_cc_removal.txt (only in example_100_100)
    │       └── zip_sizes.txt (only in example_100_100)
    ├── elections          
    │   ├── soc_positionwise_approx_cc 
    │   │   └── (empty)
    │   └── soc_original
    │       └── (800 txt files with elections)
    └── results
        ├── distances        
        │   ├── bordawise.txt
        │   └── positionwise.txt
        ├── points
        │   ├── bordawise_2d.txt
        │   └── positionwise_2d.txt
        └── winners
            └── positionwise_approx_cc.txt


Examples
=============================

Simple example of use
::

    mapel.print_2d("example_100_20", num_winners=50, winners_order="approx_cc")
    
::  

    mapel.print_2d("example_100_100", mask=True)
    
::  

    mapel.print_matrix("example_100_10", scale=0.3)


Your own (simple) experiment
-----------------------------
Imagine that you want to run your own experiment. For example you want to check wheter similar elections have the same size after compression or not. You zip all the elections from *exp_name/elections/soc_original/*. You check their sizes, and now you would like to print the map, where the *alpha* of each point is proportional to its color. 

First should normilize the values so all of them will fall into [0,1] interval. Then you should put the value with those values in *exp_name/controllers/*. One value per line -- where the first lines is corresponding with the first election and so on and so forth. If you are not sure about the format please look at *exp_name/controllers/zip_sizes.txt* file.

Finally if you would like to print the results::

    mapel.print_2d("experiment_name", values="your_file_name.txt")

For example if we run zip experiment for example_100_100 the upper line will  like this::

    mapel.print_2d("example_100_100", values="zip_sizes.txt")



Your own (complex) experiment
-----------------------------
If you want to run an experiment that is problematic time-wise and you want to run it only for a small amount of elections, we suggest you use *prepare_approx_cc_order* function to prepare the elections in approx_cc order and then run the experiment for first (for example top 50) elections from *exp_name/elections/soc_approx_cc/*. If you are chossing this option rember to set the value of *main_order* to *approx_cc*.

We do not precompute those soc files because it would have doubled the size of the package.
    
    
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
