# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os


def depth_first_search(node, visitor):
    '''
    Generic function that walks through the graph starting at ``node`` and
    uses function ``visitor`` on each node to check whether it should be
    returned.

    Args:
        node (graph node): the node to start the journey from
        visitor (Python function or lambda): function that takes a node as
         argument and returns ``True`` if that node should be returned.
    Returns:
        List of functions, for which ``visitor`` was ``True``
    '''
    stack = [node]
    accum = []
    visited = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue

        try:
            # Function node
            stack.extend(node.root_function.inputs)
        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
                    visited.add(node)
                    continue
            except AttributeError:
                pass

        if visitor(node):
            accum.append(node)

        visited.add(node)

    return accum


def find_all_with_name(node, node_name):
    '''
    Finds functions in the graph starting from ``node`` and doing a depth-first
    search.

    Args:
        node (graph node): the node to start the journey from
        node_name (`str`): name for which we are search nodes

    Returns:
        List of primitive functions having the specified name

    See also:
        :func:`~cntk.ops.functions.Function.find_all_with_name` in class
        :class:`~cntk.ops.functions.Function`.
    '''
    return depth_first_search(node, lambda x: x.name == node_name)


def find_by_name(node, node_name):
    '''
    Finds a function in the graph starting from ``node`` and doing a depth-first
    search. It assumes that the name occurs only once.

    Args:
        node (graph node): the node to start the journey from
        node_name (`str`): name for which we are search nodes

    Returns:
        Primitive function having the specified name

    See also:
        :func:`~cntk.ops.functions.Function.find_by_name` in class
        :class:`~cntk.ops.functions.Function`.

    '''
    if not isinstance(node_name, str):
        raise ValueError('node name has to be a string. You gave '
                         'a %s' % type(node_name))

    result = depth_first_search(node, lambda x: x.name == node_name)

    if len(result) > 1:
        raise ValueError('found multiple functions matching "%s". '
                         'If that was expected call find_all_with_name' % node_name)

    if not result:
        return None

    return result[0]


def plot(node, to_file):
    '''
    Walks through every node of the graph starting at ``node``,
    creates a network graph, and saves it as a string. If dot_file_name or 
    png_file_name specified corresponding files will be saved.

    Requirements:

     * for DOT output: `pydot_ng <https://pypi.python.org/pypi/pydot-ng>`_
     * for PNG output: `pydot_ng <https://pypi.python.org/pypi/pydot-ng>`_ 
       and `graphviz <http://graphviz.org>`_

    Args:
        node (graph node): the node to start the journey from
        to_file (`str`): file with either 'dot' or 'png' as suffix to denote
         what format should be written

    Returns:
        `str` containing all nodes and edges
    '''

    suffix = os.path.splitext(to_file)[1].lower()
    if suffix not in ('.png', '.dot'):
        raise ValueError('only suffix ".png" and ".dot" are supported')

    try:
        import pydot_ng as pydot
    except ImportError:
        raise ImportError(
            "PNG and DOT format requires pydot_ng package. Unable to import pydot_ng.")

    # initialize a dot object to store vertices and edges
    dot_object = pydot.Dot(graph_name="network_graph", rankdir='TB')
    dot_object.set_node_defaults(shape='rectangle', fixedsize='false',
                                 height=.85, width=.85, fontsize=12)
    dot_object.set_edge_defaults(fontsize=10)

    # string to store model
    model = []

    # walk every node of the graph iteratively
    visitor = lambda x: True
    stack = [node]
    accum = []
    visited = set()

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        try:
            # Function node
            node = node.root_function
            stack.extend(node.inputs)

            # add current node
            model.append(node.op_name)
            model.append('(')

            cur_node = pydot.Node(node.op_name + ' ' + node.uid, label=node.op_name, shape='circle',
                                  fixedsize='true', height=1, width=1)
            dot_object.add_node(cur_node)

            # add node's inputs
            for i in range(len(node.inputs)):
                child = node.inputs[i]

                model.append(child.uid)
                if i != len(node.inputs) - 1:
                    model.append(', ')

                child_node = pydot.Node(child.uid)
                dot_object.add_node(child_node)
                dot_object.add_edge(pydot.Edge(
                    child_node, cur_node, label=str(child.shape)))

            # ad node's output
            model.append(') -> ')
            model.append(node.outputs[0].uid)
            model.append('\n')

            out_node = pydot.Node(node.outputs[0].uid)
            dot_object.add_node(out_node)
            dot_object.add_edge(pydot.Edge(
                cur_node, out_node, label=str(node.outputs[0].shape)))

        except AttributeError:
            # OutputVariable node
            try:
                if node.is_output:
                    stack.append(node.owner)
            except AttributeError:
                pass

    visited.add(node)

    if visitor(node):
        accum.append(node)

    if suffix == '.png':
        dot_object.write_png(to_file, prog='dot')
    else:
        dot_object.write_raw(to_file)

    model = ''.join(model)

    return "\n".join(model.split("\n")[::-1])
