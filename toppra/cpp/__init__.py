"""Internal module for exposing cpp bindings """
try:
    from toppra.cpp.toppra_int import *
    TOPPRA_INT_LOADED = True
except ImportError:
    TOPPRA_INT_LOADED = False

def bindings_loaded():
    return TOPPRA_INT_LOADED
    
