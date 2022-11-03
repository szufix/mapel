import mapel.main.logs as logs
logger = logs.get_logger(__name__)

from mapel.allocations.metrics import idealdist

__distances = { "ideal": idealdist.ideal_distance }

def get_distance(left_task, right_task, distance_id):
    """ Return: distance between instances, (if applicable) optimal matching """
    try: 
      logger.debug(f"Getting distance: {distance_id}")
      distance_function =  __distances.get(distance_id, None)
    except KeyError:
      logger.warning(f"No distance with id: {distance_id}")
      return (0, None)
    return distance_function(left_task, right_task)


