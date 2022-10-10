import logging

from mapel.allocations.metrics import idealdist

__distances = { "ideal": idealdist.ideal_distance }

def get_distance(left_task, right_task, distance_id):
    """ Return: distance between instances, (if applicable) optimal matching """

    try: 
      logging.debug(f"Getting distance: {distance_id}")
      distance_function =  __distances.get(distance_id, None)
    except KeyError:
      logging.warning(f"No distance with id: {distance_id}")
      return (0, None)


    return distance_function(left_task, right_task)


