import os

from pymatcc import main

from utils import srcdir, cprint


if __name__ == '__main__':
    filedir = os.path.join(srcdir, 'NASICON_KMC_paper_data', 'inputs', 'prim.cif')
    cprint('pymatcc is running in test mode using the file', filedir, color='c')
    main(filedir, outdir='test')