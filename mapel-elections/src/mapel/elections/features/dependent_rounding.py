
import queue
import numpy as np


def print_tree(root):
    print(root.election_id_init, root.election_id, root.value)
    for child in root.children:
        print_tree(child)


class Node:

    def __init__(self, election_id):
        self.election_id = election_id
        self.election_id_init = election_id
        self.parent = None
        self.children = []
        self.leaf = True
        self.value = None

    def __str__(self):
        return self.election_id

    def add_child(self, child, value):
        child.parent = self
        child.value = value
        self.children.append(child)
        self.leaf = False


def _prepare_tree(vector):

    num_leaves = len(vector)

    root = Node('root')
    ctr = 0

    q = queue.Queue()
    q.put(root)

    while q.qsize() * 2 < num_leaves:
        tmp_root = q.get()
        for _ in range(2):
            inner_node = Node('v' + str(ctr))
            tmp_root.add_child(inner_node, 0)
            q.put(inner_node)
            ctr += 1

    ctr = 0
    while ctr < num_leaves:
        tmp_root = q.get()
        for _ in range(2):
            node = Node('x' + str(ctr))
            tmp_root.add_child(node, vector[ctr])
            ctr += 1

    return root


def _run_over_tree(node):
    # print(node.election_id)
    if not node.children[0].leaf:
        for child in node.children:
            _run_over_tree(child)

    # randomized_compare
    a = node.children[0].value
    b = node.children[1].value

    if a is None:
        node.value = b
        node.election_id = node.children[1].election_id

    elif b is None:
        node.value = a
        node.election_id = node.children[0].election_id

    else:

        if a + b >= 1:
            node.value = a + b - 1
            if np.random.random() > (1-a) / ((1-a)+(1-b)+0.00001):
                node.children[0].value = 1
                node.election_id = node.children[1].election_id
            else:
                node.children[1].value = 1
                node.election_id = node.children[0].election_id

        else:
            node.value = a + b
            if np.random.random() > a / (a+b+0.00001):
                node.children[0].value = 0
                node.election_id = node.children[1].election_id
            else:
                node.children[1].value = 0
                node.election_id = node.children[0].election_id

        if a+b == 1.:
            node.value = None

def _get_final_values_from_tree(node, values):
    if node.value is not None and node.election_id not in values:
        values[node.election_id] = int(node.value)
    for child in node.children:
        _get_final_values_from_tree(child, values)


def approx_rand_tree(values):
    root = _prepare_tree(values)
    _run_over_tree(root)
    values = {}
    _get_final_values_from_tree(root, values)
    return values

# values = [0.7, 0.5, 0.5, 0.5, 0.2, 0.6, 0.5, 0.5]
# root = _prepare_tree(values)
# print_tree(root)
# _run_over_tree(root)
# print("\n")
# print_tree(root)
# values = {}
# _get_final_values_from_tree(root, values)
# print(values)

