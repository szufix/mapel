#!/usr/bin/env python


from mapel.main.objects.Family import Family


class RoommatesFamily(Family):

    def __init__(self,
                 model_id: str = None,
                 family_id='none',
                 params: dict = None,
                 size: int = 1,
                 label: str = "none",
                 color: str = "black",
                 alpha: float = 1.,
                 ms: int = 20,
                 show=True,
                 marker='o',
                 starting_from: int = 0,
                 path: dict = None,
                 single_instance: bool = False,

                 num_agents: int = None):

        super().__init__(model_id=model_id,
                         family_id=family_id,
                         params=params,
                         size=size,
                         label=label,
                         color=color,
                         alpha=alpha,
                         ms=ms,
                         show=show,
                         marker=marker,
                         starting_from=starting_from,
                         path=path,
                         single_instance=single_instance)

        self.num_agents = num_agents

