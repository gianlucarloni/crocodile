from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.dirname(__file__)
parent_dir = osp.dirname(this_dir)

lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

#Matplotlib created a temporary config/cache directory at /tmp/matplotlib-6772vh08 because the default path (/.config/matplotlib) is not a writable directory; it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.
configs_path= osp.join(parent_dir,"configs")
add_path(configs_path)

pretrainedmodels_path= osp.join(parent_dir,"pretrained_models")
add_path(pretrainedmodels_path)

out_path= osp.join(parent_dir,"out")
add_path(out_path)

pycache_path= osp.join(parent_dir,"__pycache__")
add_path(pycache_path)
