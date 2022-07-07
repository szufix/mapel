
import sys

try:
    sys.path.append('/Users/szufa/PycharmProjects/abcvoting/')
    from abcvoting.preferences import Profile
    from abcvoting import abcrules, properties
    from abcvoting.output import output, INFO
except ImportError:
    pass


def test_jr(experiment, election_ids, feature_params):

    values = {}

    for rule in feature_params['rules']:
        print(rule)

        all_results = {'pareto': 0, 'jr': 0, 'pjr': 0, 'ejr': 0}

        for election_id in election_ids:

            election = experiment.elections[election_id]

            profile = Profile(election.num_candidates)
            profile.add_voters(election.votes)
            committee = election.winning_committee[rule]

            results = properties.full_analysis(profile, committee)

            if results['pareto'] == False:
                print(election_id, results)

            for name in results:
                if results[name] == False:
                    all_results[name] += 1
        print(all_results)
        values[rule] = all_results
    return values
