
import sys

try:
    sys.path.append('/Users/szufa/PycharmProjects/abcvoting/')
    from abcvoting.preferences import Profile
    from abcvoting import abcrules, properties
    from abcvoting.output import output, INFO
except ImportError:
    pass


def test_jr(experiment, election_ids, feature_params):

    for rule in feature_params['rules']:
        print(rule)

        for election_id in election_ids:

            election = experiment.elections[election_id]

            profile = Profile(election.num_candidates)
            profile.add_voters(election.votes)
            committee = election.winning_committee[rule]

            results = properties.full_analysis(profile, committee)

            # print(results.values())
            if False in results.values():
                print(results)
