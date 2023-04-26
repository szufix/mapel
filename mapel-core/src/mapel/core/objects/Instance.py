
class Instance:

    def __init__(self,
                 experiment_id: str,
                 instance_id: str,
                 culture_id: str = None,
                 alpha: float = None,
                 **kwargs):

        self.experiment_id = experiment_id
        self.instance_id = instance_id
        self.culture_id = culture_id
        self.alpha = alpha
        self.printing_params = {}

    # def __getattr__(experiment, name):
    #     if name == 'model_id':
    #         return experiment.culture_id
    #     else:
    #         try:
    #             return experiment.__dict__[name]
    #         except KeyError:
    #             raise AttributeError(name)
    #
    # def __setattr__(experiment, name, value):
    #     if name == 'model_id':
    #         experiment.culture_id = value
    #     else:
    #         experiment.__dict__[name] = value


