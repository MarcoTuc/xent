import os
import sys

class Tee(object):
    
    """ If you want to print on a txt file instead of terminal, here you have it.
        
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


def makedirs_with_check(basedir, directory):
        # Directory Check 
        path = os.path.join(basedir, directory)
        if os.path.exists(path): 
            i = 0
            while i < 4: 
                i += 1
                permission = input(f'Folder {directory} already exists, do you want to overwrite the old one? [Y,N] ')
                if permission.lower().startswith('y'):
                    print('>>>> overwriting old simulations')
                    break
                elif permission.lower().startswith('n'):
                    asknew = input(f'So, do you want to make a new folder? [Y,N] ')
                    if asknew.lower().startswith('y'):
                        while True: 
                            newdir = input(f'Write the new name:\n')
                            new_path = os.path.join(basedir, newdir)
                            if new_path != path: 
                                os.makedirs(new_path)
                                return new_path
                            else: print('Hey, put a new name not the old one... lol\n')
                    elif permission.lower().startswith('n'): 
                        print(">>>> stopping the program")
                        sys.exit()
            print(">>>> stopping the program")
            sys.exit()
        else:
            os.makedirs(directory)