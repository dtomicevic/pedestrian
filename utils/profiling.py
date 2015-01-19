from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)


def profile(f):
    """ function execution time measurement decorator

        f: function
            function being decorated
    """
    @wraps(f)
    def profiler(*args, **kwargs):
        """ measures the elapsed time for the function f and reports it
            through the logging module

            *args: list
                argument list being passed to the wrapped function

            **kwargs: dict
                argument dict being passed to the wrapped function
        """
        start = time.clock()
        res = f(*args, **kwargs)
        elapsed = time.clock() - start

        logger.info('[{0}] elapsed {1:.4f} s'.format(f.__name__, elapsed))
        return res

    return profiler
