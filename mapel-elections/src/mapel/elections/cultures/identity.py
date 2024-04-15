import prefsampling.approval as pref_approval


def identity_mask(num_voters=None,
                  num_candidates=None,
                  p=None,
                  **kwargs):
    return pref_approval.identity(num_voters=num_voters,
                                  num_candidates=num_candidates,
                                  rel_num_approvals=p,
                                  **kwargs)
