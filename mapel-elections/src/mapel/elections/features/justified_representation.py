
import sys
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    # sys.path.append(os.environ["PATH"])
    from abcvoting.preferences import Profile
    from abcvoting import abcrules, properties
    from abcvoting.output import output, INFO
except ImportError:
    pass


def test_ejr(election, feature_params):

    rule = feature_params['rule']
    print(election.election_id, rule)

    profile = Profile(election.num_candidates)
    profile.add_voters(election.votes)
    committee = election.winning_committee[rule]

    return properties.full_analysis(profile, committee)

