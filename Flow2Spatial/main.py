import os
import sys

def f2s_command():
    ''' Example of taking inputs for Flow2Spatial'''
    args = sys.argv[1:]
    if len(args) < 1:
        pass
        print("usage: generator.omics(), generator.histology() and generator.random(); model.preparation(), model.training() and model.reconstruction()")


