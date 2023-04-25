import os

def initenv(args):
    os.environ['SUNO_USE_SMALL_MODELS'] = str("-smallmodels" in args)
    os.environ['BARK_FORCE_CPU'] = str("-forcecpu" in args)
