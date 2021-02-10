
from .voting import print as pr
from .voting import canonical as can
from .voting import embedding as emb


#########################
### PREPARE ELECTIONS ###
#########################


def prepare_elections(experiment_id):
    can.prepare_elections(experiment_id)

###########################
### COMPUTING DISTANCES ###
###########################


def compute_distances_between_elections(experiment_id, **kwargs):
    can.compute_distances_between_elections(experiment_id, **kwargs)

#################
### EMBEDDING ###
#################


def convert_xd_to_2d(experiment, **kwargs):
    emb.convert_xd_to_2d(experiment, **kwargs)


################
### PRINTING ###
################

def hello():
    print("Hello!")


def print_2d(experiment_id, **kwargs):
    pr.print_2d(experiment_id, **kwargs)


def print_matrix(experiment_id):
    pr.print_matrix(experiment_id)


def print_param_vs_distance(experiment_id, **kwargs):
    pr.print_param_vs_distance(experiment_id, **kwargs)


def prepare_approx_cc_order(experiment_id, **kwargs):
    pr.prepare_approx_cc_order(experiment_id, **kwargs)
