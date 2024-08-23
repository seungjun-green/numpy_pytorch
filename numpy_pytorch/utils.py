import numpy as np

def flatten_outputs(output):
    """
    Takes any form of nested input (tensors, tuples of tensors, nested tuples of tensors)
    and returns a flat list of tensors.

    Parameters:
    - output: A tensor, tuple of tensors, or nested tuple of tensors.

    Returns:
    - A flat list containing all tensors from the nested structure.
    """
    flat_list = []
    if isinstance(output, (list, tuple)):
        for item in output:
            flat_list.extend(flatten_outputs(item))
    else:
        flat_list.append(output)

    return flat_list