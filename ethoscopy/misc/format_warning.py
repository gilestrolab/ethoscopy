def format_warning(message, category, filename, lineno, line=''):
    """
    formats warming method to not double print for user and allows string formatting
    """
    return str(filename) + ':' + str(lineno) + ': ' + category.__name__ + ': ' +str(message) + '\n'