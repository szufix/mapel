

def print_10x100_vote(experiment):
    for election in experiment.instances.values():
        election.print_map(show=True,
                           radius=25,
                           name=experiment.experiment_id,
                           alpha=0.5,
                           s=15,
                           circles=True,
                           object_type='vote',
                           title_size=30)


def print_10x1000_vote(experiment):
    for election in experiment.instances.values():
        election.print_map(show=True,
                           radius=25,
                           name=experiment.experiment_id,
                           alpha=0.5,
                           s=10,
                           circles=True,
                           object_type='vote',
                           title_size=26)


def print_10x100_candidate(experiment):
    for election in experiment.instances.values():
        election.print_map(show=True,
                           radius=1000,
                           name=experiment.experiment_id,
                           alpha=0.15,
                           s=10,
                           circles=True,
                           object_type='candidate')


def print_100x100_candidate(experiment):
    for election in experiment.instances.values():
        election.print_map(show=True,
                           radius=5000,
                           name=experiment.experiment_id,
                           alpha=0.5,
                           s=15,
                           circles=True,
                           object_type='candidate')
