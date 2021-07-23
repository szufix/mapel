
from voting import _elections as el
from voting import embedding as emb
from voting import _metrics as metr
from voting import print as pr
from voting import development as dev
from voting import matrices as mat
from voting import features


def hello():
    print("Hello!")

#####################
### ONLINE CODING ###
#####################


def generate_election(**kwargs):
    return el.generate_election(**kwargs)


def generate_experiment(**kwargs):
    return dev.generate_experiment(**kwargs)


def prepare_experiment(**kwargs):
    return dev.prepare_experiment(**kwargs)


# def generate_family(**kwargs):
#     return el.generate_family(**kwargs)


#########################
### PREPARE ELECTIONS ###
#########################





def prepare_elections(experiment_id, **kwargs):
    el.prepare_elections(experiment_id, **kwargs)


def prepare_matrices(experiment_id, **kwargs):
    mat.prepare_matrices(experiment_id, **kwargs)


###########################
### COMPUTING DISTANCES ###
###########################


def compute_distances(experiment_id, **kwargs):
    metr.compute_distances(experiment_id, **kwargs)


def compute_distances_between_votes(*args):
    return metr.compute_distances_between_votes(*args)


# def compute_distances_between_matrices(*args):
#     metr.compute_distances_between_matrices(*args)
#

#################
### EMBEDDING ###
#################


def convert_xd_to_2d(experiment, **kwargs):
    emb.convert_xd_to_2d(experiment, **kwargs)


def convert_xd_to_3d(experiment, **kwargs):
    emb.convert_xd_to_3d(experiment, **kwargs)


def embed(distances, **kwargs):
    return emb.embed(distances, **kwargs)

# def convert_using_tsne(experiment, **kwargs):
#     emb.convert_using_tsne(experiment, **kwargs)
#
# def convert_using_mds(experiment, **kwargs):
#     emb.convert_using_mds(experiment, **kwargs)


################
### PRINTING ###
################


def print_2d(experiment_id, **kwargs):
    pr.print_2d(experiment_id, **kwargs)


def print_3d(experiment_id, **kwargs):
    pr.print_3d(experiment_id, **kwargs)


def print_matrix(experiment_id, **kwargs):
    pr.print_matrix(experiment_id, **kwargs)


def print_param_vs_distance(experiment_id, **kwargs):
    pr.print_param_vs_distance(experiment_id, **kwargs)


def print_map(coordinates, **kwargs):
    pr.print_map(coordinates, **kwargs)

######################
### COMPUTE SCORES ###
######################


def compute_highest_plurality(experiment_id):
    dev.compute_highest_plurality(experiment_id)


def compute_highest_borda(experiment_id):
    dev.compute_highest_borda(experiment_id)


def compute_lowest_dodgson(experiment_id):
    dev.compute_lowest_dodgson(experiment_id)


#########################
### SUBELECTIONS ###
#########################


# def voter_subelection(experiment_id):
#     metr.voter_subelection(experiment_id)


#########################
### DEVELOPMENT ###
#########################


def get_distance(election_1, election_2, **kwargs):
    return metr.get_distance(election_1, election_2, **kwargs)


def import_election(experiment_id, election_id):
    return el.import_election(experiment_id, election_id)


def compute_subelection_weird(experiment_id, **kwargs):
    dev.compute_subelection_weird(experiment_id, **kwargs)


def compute_winners(experiment_id, **kwargs):
    dev.compute_winners(experiment_id, **kwargs)


def compute_effective_num_candidates(experiment_id, **kwargs):
    dev.compute_effective_num_candidates(experiment_id, **kwargs)


def compute_condorcet_existence(experiment_id):
    dev.compute_condorcet_existence(experiment_id)


def create_structure(experiment_id):
    dev.create_structure(experiment_id)


# def get_matrix(experiment_id, **kwargs):
#     return el.get_matrix(experiment_id, **kwargs)


# On OVERLEAF :)
def generate_votes(**kwargs):
    return el.generate_votes(**kwargs)



def generate_positionwise_matrix(**kwargs):
    return mat.generate_positionwise_matrix(**kwargs)


def get_positionwise_matrix(votes):
    return mat.get_positionwise_matrix(votes)


def compute_feature(experiment_id, name, **kwargs):
    features.compute_feature(experiment_id, name, **kwargs)