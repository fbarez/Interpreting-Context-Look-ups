def tuple_str_to_tuple(tuple_str):
    """Converts a tuple string to a tuple."""
    return tuple(map(int, tuple_str[1:-1].split(',')))