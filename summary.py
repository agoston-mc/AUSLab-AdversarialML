import os
import re
import sys

DIR = 'models'
DIR = 'vpnet/models_vpnet'

def info(filename):
    with open(f'{DIR}/{filename}', 'r') as f:
        lines = f.read()
        m = re.search(r'Test accuracy: ([\.0-9]+)%.*$', lines, re.DOTALL)
        if m is not None:
            ps = m.group(1)
            p = float(ps)
            s = f'{ps}%   {filename}'
            #if p > 94:
            s += '\n' + m.group(0).rstrip()
        else:
            p = 0
            s = f'??.??%   {filename}'
        return p, s

if __name__ == '__main__':
    files = [filename for filename in os.listdir(DIR) if filename.endswith('.log')]
    for arg in sys.argv[1:]:
        files = [filename for filename in files if arg in filename]
    data = [info(filename) for filename in files]
    data.sort(key=lambda lst: lst[0])
    print('\n'.join([x[1] for x in data]))