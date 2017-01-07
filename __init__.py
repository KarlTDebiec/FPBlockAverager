#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   fpblockaverager.__init__.py
#
#   Copyright (C) 2012-2017 Karl T Debiec
#   All rights reserved.
#
#   This software may be modified and distributed under the terms of the
#   BSD license. See the LICENSE file for details.
################################### MODULES ###################################
from __future__ import (absolute_import, division, print_function,
    unicode_literals)


################################### CLASSES ###################################
class arg_or_attr(object):
    """
    Decorator to allow obtain values either from argument or self.

    Attributes:
      names (list): Names to support
    """

    def __init__(self, *args):
        """
        Stores arguments provided at decoration.

        Arguments:
          namess (str, list): Name(s) to support
        """
        self.names = args

    def __call__(self, method):
        """
        Wraps method.

        Arguments:
          method (method): Method to wrap

        Returns:
          wrapped_function (method): Wrapped method
        """
        from functools import wraps

        decorator = self
        self.method = method

        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            """
            Wrapped version of method.

            Arguments:
              args (tuple): Arguments passed to method
              kwargs (dict): Keyword arguments passed to method

            Returns:
              return_value: Return value of wrapped method
            """

            for name in decorator.names:
                if name not in kwargs:
                    if hasattr(self, name):
                        kwargs[name] = getattr(self, name)
                    else:
                        raise ValueError(
                            "{0}.{1} could not identify argument '{"
                            "2}'.".format(
                                self.__class__.__name__,
                                decorator.method.__name__, name))

            return method(self, *args, **kwargs)

        return wrapped_method
