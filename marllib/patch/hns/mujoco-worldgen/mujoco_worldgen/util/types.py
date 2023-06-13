import functools
import inspect
import types


def enforce_is_callable(var, error_msg):
    """
    Raises an exception with provided error_msg if the variable
    is not a callable.
    """
    if not callable(var):
        raise TypeError(error_msg)
    return var


def extract_top_level_class(module, subclass):
    '''
    Searches module for a top-level class (in terms if inheritence)
    of type subclass.
    :param module: module in which we search for subclass
    :param subclass: subclass that we search for
    :return: object of type subclass.
    '''
    extracted_classes = []
    for key, value in module.items():
        if isinstance(value, types.ClassType) and issubclass(value, subclass):
            extracted_classes.append([key, value, 0])
    # Get the class which is the most top level.
    assert len(extracted_classes) > 0, "Couldn't extract %s from module: %s" % (subclass, module)
    top_level = extracted_classes[0]
    for i in range(len(extracted_classes)):
        for j in range(len(extracted_classes)):
            if issubclass(extracted_classes[i][1], extracted_classes[j][1]):
                extracted_classes[i][2] += 1
        if extracted_classes[i][2] > top_level[2]:
            top_level = extracted_classes[i]
    return top_level[1]


def extract_matching_arguments(fun, kwargs):
    # Extracts subset of kwargs that contains arguments present in signature of fun.
    assert callable(fun), "First argument to extract_matching_arguments should be a function."
    assert isinstance(kwargs, dict), "Second argument to extract_matching_arguments should be a dictionary of arugments"
    fun_handler = fun
    while hasattr(fun_handler, "__wrapped__") or inspect.isclass(fun_handler):
        if hasattr(fun_handler, "__wrapped__"):
            fun_handler = fun_handler.__wrapped__
        if inspect.isclass(fun_handler):
            fun_handler = fun_handler.__init__
    spec = inspect.getfullargspec(fun_handler)
    fun_args = []
    if spec.args is not None:
        fun_args += spec.args
    if spec.kwonlyargs is not None:
        fun_args += spec.kwonlyargs
    # function accepts kwargs. Therefore, we pass all arguments.
    if spec.varkw is not None:
        args_to_pass = kwargs
        args_remaining = {}
    else:
        args_to_pass = dict([(k, v) for k, v in kwargs.items() if k in fun_args])
        args_remaining = dict([(k, v) for k, v in kwargs.items() if k not in fun_args])
    return args_to_pass, args_remaining


def store_args(method):
    """Stores provided method args as instance attributes."""
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        list(
            map(args.update, (zip(arg_names, positional_args[1:]), keyword_args.items())))
        # Store values in instance as attributes
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


class Maybe(type):
    ''' Metaclass to match types with optionally none.  Use maybe() instead '''
    maybe_type = type(None)  # Overridden in derived classes

    def __instancecheck__(self, instance):
        return isinstance(instance, self.maybe_type) or instance is None

    def __repr__(self):
        return "<class Maybe({})>".format(self.maybe_type)


def maybe(arg_type):
    '''
    Helper for @accepts and @returns decorator.  Maybe means optionally None.
    Example:
        @accepts(maybe(int), str, maybe(dict))
        def foo(a, b, c):
            # a - can be int or None
            # b - must be str
            # c - can be dict or None
    See: https://wiki.haskell.org/Maybe
    '''
    class Derived(metaclass=Maybe):
        maybe_type = arg_type
    return Derived


# Copied from
# http://pythoncentral.io/validate-python-function-parameters-and-return-types-with-decorators/
def accepts(*accepted_arg_types):
    '''
    A decorator to validate the parameter types of a given function.
    It is passed a tuple of types. eg. (<type 'tuple'>, <type 'int'>)

    Note: It doesn't do a deep check, for example checking through a
          tuple of types. The argument passed must only be types.

    See also the maybe(), used for types that are optionally None.
    '''

    def accept_decorator(validate_function):
        ''' Do not call this function directly!  Use @accepts(...) instead! '''
        # Check if the number of arguments to the validator
        # function is the same as the arguments provided
        # to the actual function to validate. We don't need
        # to check if the function to validate has the right
        # amount of arguments, as Python will do this
        # automatically (also with a TypeError).
        @functools.wraps(validate_function)
        def decorator_wrapper(*function_args, **function_args_dict):
            if len(accepted_arg_types) is not len(accepted_arg_types):
                raise InvalidArgumentNumberError(validate_function.__name__)

            # We're using enumerate to get the index, so we can pass the
            # argument number with the incorrect type to
            # ArgumentValidationError.
            for arg_num, (actual_arg, accepted_arg_type) in enumerate(zip(function_args, accepted_arg_types)):
                if not isinstance(actual_arg, accepted_arg_type):
                    ord_num = _ordinal(arg_num + 1)
                    raise ArgumentValidationError(ord_num,
                                                  validate_function.__name__,
                                                  accepted_arg_type)
            return validate_function(*function_args, **function_args_dict)

        return decorator_wrapper

    return accept_decorator


def returns(*accepted_return_type_tuple):
    '''
    Validates the return type. Since there's only ever one
    return type, this makes life simpler. Along with the
    accepts() decorator, this also only does a check for
    the top argument. For example you couldn't check
    (<type 'tuple'>, <type 'int'>, <type 'str'>).
    In that case you could only check if it was a tuple.

    See also maybe() for optionally returning a type or None
    '''

    def return_decorator(validate_function):
        ''' Do not call this function directly!  Use @returns(...) instead ! '''
        # No return type has been specified.
        if len(accepted_return_type_tuple) == 0:
            raise TypeError('You must specify a return type.')

        @functools.wraps(validate_function)
        def decorator_wrapper(*function_args, **function_args_dict):
            # More than one return type has been specified.
            if len(accepted_return_type_tuple) > 1:
                raise TypeError('You must specify one return type.')

            # Since the decorator receives a tuple of arguments
            # and the is only ever one object returned, we'll just
            # grab the first parameter.
            accepted_return_type = accepted_return_type_tuple[0]

            # We'll execute the function, and
            # take a look at the return type.
            return_value = validate_function(*function_args, **function_args_dict)
            return_value_type = type(return_value)

            if isinstance(return_value_type, accepted_return_type):
                raise InvalidReturnType(return_value_type,
                                        validate_function.__name__)

            return return_value

        return decorator_wrapper

    return return_decorator


def _ordinal(num):
    '''
    Returns the ordinal number of a given integer, as a string.
    eg. 1 -> 1st, 2 -> 2nd, 3 -> 3rd, etc.
    '''
    if 10 <= num % 100 < 20:
        return '{0}th'.format(num)
    else:
        ord = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
        return '{0}{1}'.format(num, ord)


class ArgumentValidationError(ValueError):
    '''
    Raised when the type of an argument to a function is not what it should be.
    '''

    def __init__(self, arg_num, func_name, accepted_arg_type):
        self.error = 'The {0} argument of {1}() is not a {2}'.format(arg_num,
                                                                     func_name,
                                                                     accepted_arg_type)

    def __str__(self):
        return self.error


class InvalidArgumentNumberError(ValueError):
    '''
    Raised when the number of arguments supplied to a function is incorrect.
    Note that this check is only performed from the number of arguments
    specified in the validate_accept() decorator. If the validate_accept()
    call is incorrect, it is possible to have a valid function where this
    will report a false validation.
    '''

    def __init__(self, func_name):
        self.error = 'Invalid number of arguments for {0}()'.format(func_name)

    def __str__(self):
        return self.error


class InvalidReturnType(ValueError):
    '''
    As the name implies, the return value is the wrong type.
    '''

    def __init__(self, return_type, func_name):
        self.error = 'Invalid return type {0} for {1}()'.format(return_type,
                                                                func_name)

    def __str__(self):
        return self.error
