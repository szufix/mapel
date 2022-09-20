
class Instance:

    def __init__(self, experiment_id: str, instance_id: str, culture_id: str = None,
                 alpha: float = None):

        self.experiment_id = experiment_id
        self.instance_id = instance_id
        self.culture_id = culture_id
        self.alpha = alpha

    def __getattr__(self, name):
        if name == 'model_id':
            return self.culture_id
        else:
            try:
                return self.__dict__[name]
            except KeyError:
                raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == 'model_id':
            self.culture_id = value
        else:
            self.__dict__[name] = value


# # # # # # # # # # # # # # # #
# LAST CLEANUP ON: 22.10.2021 #
# # # # # # # # # # # # # # # #
