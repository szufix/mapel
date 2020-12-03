Introduction
=============================
**Mapel** (map of elections) is a python package that serves for drawing maps of elections. It contains a testbed of elections to be used
for various election-related experiments such as testing algorithms or estimating the frequency of a given phenomenon. If you want to run an experiment on a large variety of elections then mapel is for you!

For more details please look at:

    Szufa,  S.,  Faliszewski,  P.,  Skowron,  P.,  Slinko,  A.,  Talmon,  N.:  Drawing  a  map of elections in the space of statistical cultures. In: Proceedings of AAMAS-2020, 1341-1349.


The package contains the following elements

* several sets of elections, including the one used by Szufa et al. [AAMAS-2020]
* tools for drawing maps of elections
* tools for generating elections and computing distances between them (*available soon*)

**Citation Policy**: if you are using this package, we kindly ask you to cite the above article *Drawing  a  map of elections in the space of statistical cultures*.

Installation
-----------------------------
For mapel to work you need **python 3.6** or newer

To install mapel on your system type::

    pip install mapel


Note that this will install necessary dependencies  (e.g. matplotlip, numpy, etc).

Beside installing the package please download mapel_data.zip file from https://github.com/szufix/mapel_data/
which contains all the data. After downloading extract it to wherever you want to use the package.

After extracting the structure should look as follows::

    your_folder
        ├── experiments/
        ├── images/
        └── test.py


Testing
-----------------------------
Inside mapel_data there as a python file *test.py*::

    import mapel
    
    mapel.hello()
    
    # mapel.print_2d("testbed_100_100", mask=True)

If everything was correctly downloaded and imported then after running test.py you should see "Welcome to Mapel!" text.

As a next step you can uncomment the last line and after running test.py, it should display the map of elections.

If you have any problems so far please contact: *stanislaw.szufa@uj.edu.pl*

Features
-----------------------------
Here we present main features of the mapel package.

* printing maps of elections
* printing matrices of distances
* generating elections according to many models (*available soon*)
* computing distances between elections (*available soon*)



Examples
-----------------------------
Simple examples of use. Just type the following commands in python and enjoy the results.


::

    mapel.print_2d("testbed_100_100", values="hb_time", mask=True)
    
::

    mapel.print_2d("testbed_100_20")
    
::  

    mapel.print_matrix("testbed_100_10", scale=0.3)
    
Experiments
=============================
The mapel package contains 6 precomputed experiments. Each of them contains a mixture of 800 election from 30 different  models: 

- Impartial Culture, Single Crossing, SPOC, Single Peaked (by Walsh), Single Peaked (by Conitzer),
- Euclidean: 1D Interval, 2D Square, 3D Cube, 5D Cube, 10D Cube 20D Cube, 2D Sphere, 3D Sphere, 5D Sphere,  
- Urn Model with the following parameter: 0.5, 0.2, 0.1, 0.05, 0.02, 0.01 
- Mallows with the following parameter: 0.999, 0.99, 0.95, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.001

Each of these experiments regards 100 voters and x candidates where x = 3,4,5,10,20,100.

The names of these experiments are:

- Experiment 1: 100 voters, 100 candidates; exp_name: **testbed_100_100**
- Experiment 2: 100 voters, 20 candidates; exp_name: **testbed_100_20**
- Experiment 3: 100 voters, 10 candidates; exp_name: **testbed_100_10**
- Experiment 4: 100 voters, 5 candidates; exp_name: **testbed_100_5**
- Experiment 5: 100 voters, 4 candidates; exp_name: **testbed_100_4**
- Experiment 6: 100 voters, 3 candidates; exp_name: **testbed_100_3**
    
General structure of a single experiment::

    ?exp_name?
    ├── controllers     
    │   ├── advanced/ 
    │   ├── basic/ 
    │   ├── distances/ 
    │   ├── orders/
    │   └── points/
    ├── elections
        └── soc_original/

            
Exact structure of percomputed experiments::

    ?exp_name?
    ├── controllers     
    │   ├── basic
    │   │   ├── map.txt                                     #  contains all technical details of the experiment
    │   │   └── matrix.txt                                  #  auxiliary file print_matrix() function
    │   └── advanced
    │       ├── hb_time.txt (only in testbed_100_100)
    │       └── zip_sizes.txt (only in testbed_100_100)
    ├── elections          
    │   ├── soc_positionwise_approx_cc 
    │   │   └── (empty)
    │   └── soc_original
    │       └── (800 txt files with elections)                  #  all the elections -- each election in a separate file
    └── results
        ├── distances        
        │   ├── bordawise.txt (only in testbed_100_100)         #  bordawise distances between each pair of elections
        │   └── positionwise.txt                                #  positionwise distances between each pair of elections
        ├── orders
        │   └── positionwise_approx_cc.txt                      #  ranking of elections
        ├── points
        │   ├── bordawise_2d.txt (only in testbed_100_100)      #  coordinates of embedded points
        │   └── positionwise_2d.txt                             #  coordinates of embedded points
        └── scores
            ├── highest_borda (only in testbed_100_100)
            ├── highest_copeland (only in testbed_100_100)
            ├── highest_dodgson (only in testbed_100_100)
            └── highest_plurality (only in testbed_100_100)

You can your own experiments, but remember that they should have the same structure. If you want to create an experiment of your own we suggest you first copy one of the existing experiemnts and then just replace necessary files.

Controllers are described in details in the last section.


Advanced example of use (1)
-----------------------------
Imagine that you want to check whether similar elections have the same size after compression or not. You zip all the elections from *?exp_name?/elections/soc_original/*. You check their sizes, and now you would like to print the map.

You should put the file with those values in *?exp_name?/controllers/advanced*. One value per line -- where the first line is corresponding to the first election, the second one corresponds to the second election and so on and so forth. If you are not sure about the format, please look at *?exp_name?/controllers/advanced/zip_size.txt* file.

Let us assume that you run your experiment for testbed_100_100. If you want to print a map, you just need to type::

    mapel.print_2d("testbed_100_100", values="zip_size", mask=True)
    
If you want to use different coloring we recommend using cmap::

    my_cmap = mapel.custom_div_cmap(colors=["white", "orange", "red"])
    mapel.print_2d("testbed_100_100", values="zip_size", mask=True, cmap=my_cmap)
    
More detailed description of all the parameters can be found in the next section called *Functionalities*. 

If we would like to see the correlation of zip_sizes and the average distance from IC elections, we should type::

    mapel.print_param_vs_distance("testbed_100_100", values="zip_size")


Representative set of elections
-----------------------------
800 elections is really a lot, and many elections within those 800 are very similar to one another. The basic idea is that we wanted to create a smaller set that will be representative. By representative set of elections we mean such set that by testing some algorithm on this set we will draw more or less the same conclusions as while testing that algorithm  on all 800 elections.

Using approximation algorithm for Chamberlin-Courant voting rule, we precomputed a ranking of all 800 elections. Each election was a voter ana a candidate at the same time. The smaller was the (positionwise) distance between two elections the higher they appear in one another vote. We refer to this ranking as *approx_cc*.


Advanced example of use (2)
-----------------------------
If you want to test an algorithm that is taking a lot of time to compute and you want to run it only on few elections, we suggest that you use *prepare_approx_cc_order* function to prepare the elections in approx_cc order and then run the experiment for first (for example top 200) elections from *?exp_name?/elections/soc_?metric?_approx_cc/*. If you are choosing  this option, remember to set the value of *order* to *?metric?_approx_cc*.



Functionalities
=============================
In this section we describe in details the functionalities of mapel.

Printing the map of elections
-----------------------------
**print_2d** function is displaying a two dimensional embedding of all the elections from a given experiment.
::

    mapel.print_2d(experiment_id, mask=False,
             angle=0, reverse=False, values=None,
             num_elections=800, main_order_name="default", metric="positionwise",
             saveas="map_2d", show=True, ms=9, normalizing_func=None, xticklabels=None, cmap='Purples_r',
             ignore=None, marker_func=None, tex=False, black=False) 

experiment_id
  : obligatory, string; name of the experiment.
  
num_elections
  : optional, int, number of points to be printed.
  
main_order_name
  : optional, string; name of the file that contains the order in which the points should appear.
  
values
  : optional, string; name of the file that contains 'values'. The file should be in *?exp_name?/controllers/advanced/* folder.

cmap
  : optional [use only if 'values is not None], cmap; use cmap coloring.

normalizing_func
  : optional [use only if 'values is not None], function;  marker_func takes single argument 'float' and returns 'float'.

xticklabels
  : optional [use only if 'values is not None], list[]; define xtick labels.

marker_func=None,
  : optional [use only if 'values is not None], function; marker_func takes single argument 'float' and returns 'marker' (e.g., 'x').
  
angle
  : optional, float; rotate the image by *angle*.
  
reverse
  : optional, bool; reverse the image.
    
mask
  : optional, bool; mark all families on the map (only for *testbed_100_100*).".
  
metric
  : optional, string; name of the metric.
  
saveas
  : optional, string; name of the saved file.
  
show
  : optional, bool; if set to False the results will not be displayed.
  
ms 
  : optional, int; marker size.

ignore 
  : optional, list[int]; list containg ids of election to ignore (not print).
  
tex
  : optional, bool; save file in a tex format.
  
black
  : optional, bool; only for mask: print names in black.

Printing the matrix with distances
-----------------------------
**print_matrix** function is displaying an array with average distances between each family of elections from a given experiment.

::

    mapel.print_matrix(experiment_id, scale=1., metric="positionwise", saveas="matrix", show=True)

experiment_id
  : obligatory, string; name of the experiment.
  
scale
  : optional, string; multiply all the values by *scale*.
   
metric
  : optional, string; name of the metric.
  
saveas
  : optional, string; name of the saved file.
  
show
  : optional, bool, if set to False the results will not be displayed.


Printing the plot of a given election parameter against the average distance from IC.
-----------------------------
**print_param_vs_distance** function is printing an array with average distances between each family of elections from a given experiment. For now, it works only with original testbed_100_100.

::

    mapel.print_param_vs_distance(experiment_id, values="", scale="none", metric="positionwise", saveas="correlation", show=True)

experiment_id
  : obligatory, string; name of the experiment.
  
values
  : obligatory, string; name of the file that contains param values. The file should be in *?exp_name?/controllers/advanced/* folder.
  
scale
  : optional, string; scale your param values with "log" or "loglog".
  
metric
  : optional, string; name of the metric.
 
saveas
  : optional, string; name of the saved file.
  
show
  : optional, bool, if set to False the results will not be displayed.


Prepare SOC files
-----------------------------
**prepare_approx_cc_order** function serves for preparing elections in soc format in approx_cc order. This function is just coping files from *soc_original* and pasting them in an order from *?exp_name?/results/orders/?metric?_approx_cc.txt*. 

::

    mapel.prepare_approx_cc_order(experiment_id, metric="positionwise")

experiment_id
  : obligatory; name of the experiment.
 
metric
  : optional, string; name of the metric.
  
  
Create CMAP
-----------------------------
**custom_div_cmap** function serves for creating cmap.

::

    custom_div_cmap(colors=None, num_colors=101)

colors
  : optional, List[string]; list of leading colors (e.g., ['white', 'orange', 'red'])
 
num_colors
  : optional, string; number of different colors (it does not have to do anything with the length of the upper list).
  
       
       
Tutorial
=============================
In this section we show how to conduct the whole experiment from the very beginning till the end.

1) Install the package:

    pip install mapel
    
2) Download the data from https://github.com/szufix/mapel_data/ and extract it wherever you want.
3) Test the *import* by running the test.py file, that is located inside the mapel_data folder. It should print "*Welcom to Mapel*" text.
4) Test the *data* by uncommenting last line in the test.py file, and then running the file again. You should see the main map containg 800 points (elections).
5) Compute Borda score for each elections. You can use the code presented below.

::

    import os
    from mapel.voting import objects as obj

    def get_highest_borda_score(election):
    scores = [0 for _ in range(election.num_candidates)]
    for vote in election.votes:
        for i in range(len(vote)):
            scores[vote[i]] += election.num_candidates - i - 1
    return max(scores)


    def compute_highest_borda_map(experiment_id):

        model = obj.Model(experiment_id)

        for i in range(model.num_elecitons):
            election_id = 'core_' + str(i)
            election = obj.Election(experiment_id, election_id)

            score = get_highest_borda_score(election)
            print(i, score)

            file_name = 'borda_score.txt'
            path = os.path.join(os.getcwd(), 'experiments', experiment_id, 'controllers', 'advanced', file_name)
            with open(path, 'a') as txtfile:
                txtfile.write(str(score) + "\n")


    if __name__ == "__main__":

        experiment_id = 'testbed_100_100'
        compute_highest_borda_map(experiment_id)

6) If you computed borda scores on your own rember to put them in experiments/*experiment_id*/controllers/advanced/*file_name*.txt
7) Run the following command:

::

   import mapel 
   experiment_id = 'testbed_100_100'
   file_name = 'borda_score'
   mapel.print_2d(experiment_id, values=file_name)   
    
8) Enjoy the results!

    
    




    
Extras
=============================

Controllers
-----------------------------
The whole technical description of an experiment is kept in *?exp_name?/controllers/basic/map.txt". 

Before editing this file, please make a safe copy. The content looks as follows::

    number_of_voters

    number_of_candidates

    number_of_families

    first_family_size, first_family_code, first_family_param_1,  first_family_param_2, first_family_color, first_family_alpha, first_family_label

    second_family_size, second_family_code, second_family_param_1, second_family_param_2, second_family_color, second_family_alpha, second_family_label

    ...

    last_family_size, last_family_code, last_family_param_1, last_family_param_2, last_family_color, last_family_alpha, last_family_label
    
    
Detailed explanation

* size -- number of elections from a given family
* code -- the id of the election model, for example impartial_culture, 3d_sphere or 20d_cube
* param -- model's parameter; only important urn_model or mallows
* color -- the color in which the family will be displayed
* alpha -- transparency
* label -- full name of the family; for example "Urn Model 0.1"

If you want to hide a given family and do not print it, just put '#' at the begging of a that family line::

    #that_family_size, that_family_code, that_family_param, that_family_color, that_family_alpha, that_family_label


Matrix with distances
-----------------------------
If you want to print just several selected families of elections or change the order in which they appear, you should go to the file:  "*?exp_name?/controllers/basic/matrix.txt*". There is list of names of all the families of elections. The number of families and their order can be change and will influence the *mapel.print_matrix()* function.

SOC files
-----------------------------
Definition of the soc format can be found here: http://www.preflib.org/data/format.php#soc



Contact
=============================
If you have any questions or have found a bug please email me at *stanislaw.szufa@uj.edu.pl*
