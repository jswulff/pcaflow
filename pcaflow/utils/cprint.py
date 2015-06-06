#! /usr/bin/env python2

import sys

def cprint(text,params):
    """
    Conditional print, based on params['debug']

    """
    if params['debug'] > 0:
        sys.stdout.write(text)
        sys.stdout.write('\n')
        sys.stdout.flush()
