import math
scalinglaws_lr_function = lambda n: 0.003239 - (0.0001395 * math.log(n))

class Tee(object):
    """ So you want to print on a txt file instead of terminal, here you have it.
        
        > How to use it
            import sys
            f = open(os.path.join(path_to_use, "console.txt"), "w+")
            sys.stdout = Tee(f) 
    """
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self):
        for f in self.files:
            f.flush()
    def close(self):
        for f in self.files:
            f.close()