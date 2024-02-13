import prefsampling.approval as pref_approval
import prefsampling.ordinal as pref_ordinal


def truncated_urn_mask(num_voters=None,
                       num_candidates=None,
                       p=None,
                       alpha=None,
                       **kwargs):
    return pref_approval.truncated_ordinal(num_voters=num_voters,
                                           num_candidates=num_candidates,
                                           p=p,
                                           ordinal_sampler=pref_ordinal.urn,
                                           ordinal_sampler_parameters={'alpha': alpha})
