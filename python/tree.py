import random
import torch

NODE_START_TIMES = []
NODE_END_TIMES = []
LINEAR_REMAINING = 0
RANDOM_REMAINING = 0

class Node:
    def __init__(self):
        self.children = []
        self.start_time = None
        self.end_time = None

def add_random_nodes(root):
    # print(linear_remaining, random_remaining)
    global LINEAR_REMAINING, RANDOM_REMAINING
    if LINEAR_REMAINING == 0 and RANDOM_REMAINING == 0:
        return
    if LINEAR_REMAINING > 0:
        new_node = Node()
        root.children.append(new_node)
        # print("Linear Node: ", LINEAR_REMAINING, RANDOM_REMAINING)
        LINEAR_REMAINING -= 1
        add_random_nodes(new_node)
    else:
        num_children = random.randint(1, min(RANDOM_REMAINING, 10))
        RANDOM_REMAINING -= num_children
        # print("Random Node: ", LINEAR_REMAINING, RANDOM_REMAINING, num_children)
        for _ in range(num_children):
            new_node = Node()
            root.children.append(new_node)
            add_random_nodes(new_node)

def generate_tree():
    global LINEAR_REMAINING
    root = Node()
    LINEAR_REMAINING -= 1
    add_random_nodes(root)
    return root


def dfs_(node, time):
    if node is not None:
        node.start_time = time
        time += 1
        for child in node.children:
            time = dfs_(child, time)
        node.end_time = time
        time += 1
    return time

def collect_node_times(node):
    global NODE_START_TIMES, NODE_END_TIMES
    if node is not None:
        NODE_START_TIMES.append(node.start_time)
        NODE_END_TIMES.append(node.end_time)
        for child in node.children:
            collect_node_times(child)

def create_causal_mask():
    num_nodes = len(NODE_START_TIMES)
    causal_mask = [[False] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(num_nodes):
            if NODE_START_TIMES[i] <= NODE_START_TIMES[j] and NODE_END_TIMES[i] >= NODE_END_TIMES[j]:
                # Transpose of causal mask to get lower triangular matrix
                causal_mask[j][i] = True
    return causal_mask

def dfs(root):
    global NODE_START_TIMES, NODE_END_TIMES
    NODE_START_TIMES = []
    NODE_END_TIMES = []
    dfs_time = 0
    dfs_(root, dfs_time)
    collect_node_times(root)
    causal_mask = create_causal_mask()
    return NODE_START_TIMES, NODE_END_TIMES, causal_mask


def generate_random_trees(num_trees, num_nodes, root_chain):
    global LINEAR_REMAINING, RANDOM_REMAINING
    tree_start_times = []
    tree_end_times = []
    causal_mask = []
    for _ in range(num_trees):
        LINEAR_REMAINING = root_chain
        RANDOM_REMAINING = num_nodes
        root = generate_tree()
        _start_time, _end_times, _causal_mask = dfs(root)
        assert len(_start_time) == num_nodes + root_chain
        assert len(_end_times) == num_nodes + root_chain
        tree_start_times.append(_start_time)
        tree_end_times.append(_end_times)
        causal_mask.append(_causal_mask)

    start_times = torch.tensor(tree_start_times, dtype=torch.float32).cuda()
    end_times = torch.tensor(tree_end_times, dtype=torch.float32).cuda()
    causal_masks = torch.tensor(causal_mask).cuda()

    return start_times, end_times, causal_masks

if __name__ == "__main__":
    tree_start_times, tree_end_times, causal_mask = generate_random_trees(1, 10, 3)
    print(tree_start_times)
    print(tree_end_times)
    print(causal_mask)