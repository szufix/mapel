
from .voting import elections as el
from .voting import metrics as metr
from .voting import embedding as emb
from .voting import print as pr
from .voting import advanced as adv


def hello():
    print("Hello!")


#########################
### PREPARE ELECTIONS ###
#########################


def prepare_elections(experiment_id):
    el.prepare_elections(experiment_id)

###########################
### COMPUTING DISTANCES ###
###########################


def compute_distances(experiment_id, **kwargs):
    metr.compute_distances(experiment_id, **kwargs)


#################
### EMBEDDING ###
#################


def convert_xd_to_2d(experiment, **kwargs):
    emb.convert_xd_to_2d(experiment, **kwargs)


def convert_xd_to_3d(experiment, **kwargs):
    emb.convert_xd_to_3d(experiment, **kwargs)


################
### PRINTING ###
################


def print_2d(experiment_id, **kwargs):
    pr.print_2d(experiment_id, **kwargs)


def print_3d(experiment_id, **kwargs):
    pr.print_3d(experiment_id, **kwargs)


def print_matrix(experiment_id):
    pr.print_matrix(experiment_id)


def print_param_vs_distance(experiment_id, **kwargs):
    pr.print_param_vs_distance(experiment_id, **kwargs)



######################
### COMPUTE SCORES ###
######################


def compute_highest_plurality(experiment_id):
    adv.compute_highest_plurality(experiment_id)


def compute_highest_borda(experiment_id):
    adv.compute_highest_borda(experiment_id)