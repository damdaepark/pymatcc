import os
import sys
import signal
import errno
import itertools
import time
import pickle
from datetime import datetime
from inspect import stack
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import colors
import colorcet as cc  # pip install colorcet

pd.options.mode.chained_assignment = None

# Figure settings
fs = 12
dpi = 200
config_figure = {'figure.figsize': (5,4), 'figure.titlesize': fs,
                 'font.size': fs, 'font.family': 'sans-serif', 
                 'font.serif': ['computer modern roman'], 
                 'font.sans-serif': ['Avenir LT Std'],
                 'axes.xmargin': 0, 'axes.titlesize': fs, 'axes.labelsize': fs, 'axes.labelpad': 0, 
                 'xtick.labelsize': fs, 'ytick.labelsize': fs, 'xtick.major.pad': 0, 'ytick.major.pad': 0, 
                 'legend.fontsize': fs, 'legend.title_fontsize': fs,
                 'lines.linewidth': 1, 'figure.dpi': dpi, 'savefig.dpi': dpi*2,
                 'text.usetex': False, 'mathtext.default': 'regular', 
                 'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm,physics,lmodern,cmbright}'}
mpl.rcParams.update(config_figure)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def list_fonts():
    fpaths = fm.findSystemFonts()

    for fpath in fpaths:
        try:
            f = fm.get_font(fpath)
        except:
            print('Error occurred in', fpath)
        print(f.family_name)
    return


def isstring(value):
    return isinstance(value, str)


def isnumeric(value):
    return isinstance(value, (float, int, np.number)) and not np.isnan(value)

def isscalar(value):
    return isnumeric(value)


def isarray(value):
    return isinstance(value, (list, tuple, np.ndarray))


def istuple(value):
    return isinstance(value, tuple)


def isstringdata(data):
    return any(isstring(x) for x in data)


def isnumericdata(data):
    return any(isnumeric(x) for x in data)

def isscalardata(data):
    return isnumericdata(data)


def isarraydata(data):
    return any(isarray(x) for x in data)


def istupledata(data):
    return any(istuple(x) for x in data)


def isscalarlist(data):
    lengths = calc_length(data)
    if isarraydata(data) and len(lengths.unique()) == 1 and lengths[0] == 1:
        return True
    else:
        return False


def apply_specialc(carray, specialc, specialc_loc, X):
    scolor = np.array(list(colors.to_rgba(specialc))).reshape(1, -1)
    if specialc_loc == 'first':
        loc = (X == np.argmin(X))
    else:
        loc = (X == np.argmax(X))
    carray[loc] = scolor
    return carray


def custom_linear_cmap(X):
    norm = mpl.colors.Normalize(vmin=np.min(X), vmax=np.max(X))
    values = np.linspace(min(X), max(X), 6)
    colors = [(255, 255, 255), (252, 223, 3), (252, 200, 4), (252, 178, 6), (252, 135, 10), (239, 3, 23)]
    carray = [(norm(value), tuple(np.array(color) / 255)) for value, color in zip(values, colors)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('', carray)
    return cmap


def get_carray(X, mapping='linear', palette='winter', alpha=1, specialc=False, specialc_loc='first'):
    # Parse input
    if isinstance(X, int):  # number is given
        N = X
        X = list(range(N))
        mapping = 'equal'  # override
    else:
        N = len(X)
    
    if palette == 'custom_linear':
        cmap = custom_linear_cmap(X)
    else:
        cmap = plt.get_cmap(palette)
        
    # Assign color
    if mapping == 'equal':  # equally spaced
        carray = cmap(np.linspace(0, 1, N))
        if specialc:
            carray = apply_specialc(carray, specialc, specialc_loc, X)
        indices = np.argsort(X)
        carray = carray[indices,:]
    elif mapping == 'linear':  # linearly mapped
        norm = plt.Normalize()
        carray = cmap(norm(X))
        if specialc:
            carray = apply_specialc(carray, specialc, specialc_loc, X)
    else:
        raise('#TODO')
    
    # Transparency
    carray[:, -1] = alpha
    return carray


def set_colorbar(fig, pos, values, palette='winter', label=None, 
                 orientation='vertical', remove_ticks=False, labelpad=2, 
                 fontsize=10, shrink=1):
    # Set colormap
    if palette == 'custom_linear':
        cmap = custom_linear_cmap(values)
    else:
        cmap = plt.get_cmap(palette)
    norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    
    # Set location
    cax = fig.add_axes(pos)
    cbar = fig.colorbar(mappable=mappable, cax=cax, orientation=orientation, shrink=shrink)
    if orientation == 'horizontal':
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('bottom')
        
    # Decoration
    if remove_ticks:
        cbar.set_ticks([])
    if label:
        if orientation == 'horizontal':
            cbar.ax.set_xlabel(label, labelpad=labelpad, fontsize=fontsize)
        else:
            cbar.ax.set_ylabel(label, labelpad=labelpad, fontsize=fontsize)
    return cbar
    

def inspect_missing_values(df):
    num_inf = np.isinf(df).values.sum()
    num_nan = np.isnan(df).values.sum()
    num_null = df.isnull().values.sum()
    if num_inf + num_nan + num_null > 0:
        raise('Some non-numeric values were detected')
    else:
        return
    
    
def timer(i, N=None):
    while 1:
        if timestamp(only_hour=True) >= 18 or \
            timestamp(only_hour=True) < 9 or \
            datetime.today().strftime('%A') in ['Saturday', 'Sunday']:
            return
        else:
            if N:
                string = '(' + str(i) + '/' + str(N) + ')'
            else:
                string = i
            cprint('Finished up to', string, '. Sleep 1 hour...')
            time.sleep(60*60)  # 1 hour sleep
    return


def timestamp(formal=False, only_hour=False):
    if only_hour:
        return int(time.strftime('%H'))
    elif formal:
        return time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return time.strftime('%y%m%d_%H%M%S')
    return


def remove_axes(ax):
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return


def find_factors(number):
    factors = list()
    for i in range(1, number + 1):
        if number % i == 0:
            factors.append(i)
    return factors


def make_rectangular(array):  # working for list and array
    if len(array) <= 4:
        array = np.atleast_2d(array)
        m, n = array.shape
        return array, m, n
    
    N = len(array)
    if N % 2 != 0:
        if type(array) is list:
            array.append(None) 
        elif type(array) is np.ndarray:
            array = np.append(array, None)
        else:
            raise('#TODO')
        N += 1
    array = np.atleast_2d(array)
    factors = find_factors(N)
    for m in np.flipud(factors):
        n = int(N/m)
        if m/n < 1:
            array = array.reshape(m, n)
            return array, m, n


def overrite_folder(source_folder, destination_folder, ignore=None):
    for file_name in os.listdir(source_folder):
        if file_name in ignore:
            continue
        source = os.path.join(source_folder, file_name)
        destination = os.path.join(destination_folder, file_name)
        shutil.copy(source, destination)
        

def detect_root(path, debug=False):
    try:
        # folderlist = [d for d in os.listdir(path) if os.path.isdir(d)]
        folderlist = [d for d in os.listdir(path)]
        if debug:
            print('Files inside ', path)
            for d in os.listdir(path):
                print(d + ' is ' + str(os.path.isdir(d)))
            print('')
        
        if 'script' in folderlist:
            return True
        else:
            return False
    except:
        return False
        
        
def find_root(path):
    max_depth = 5
    depth = 0
    while 1:
        path = os.path.abspath(os.path.join(path, os.pardir))   
        if detect_root(path):
            break
        elif depth > max_depth:
            raise('Cannot find root.')
        depth += 1
    return path


def set_path(workingdir):
    rootdir = find_root(workingdir)
    datadir = os.path.join(rootdir, 'dat')
    scriptdir = os.path.join(rootdir, 'script')
    imgdir = os.path.join(rootdir, 'img')
    srcdir = os.path.join(rootdir, 'src')
    sys.path.append(scriptdir)
    sys.path.append(srcdir)
    return workingdir, rootdir, datadir, scriptdir, imgdir, srcdir
workingdir, rootdir, datadir, scriptdir, imgdir, srcdir = set_path(os.path.abspath(__file__))


def flatten_list(l):
    if not type(l[0]) is list:
        cprint('Given list is not a nested list. Skip.')
        return l
    try:
        return list(itertools.chain(*l))
    except:
        out = []
        _l = l
        while _l:
            x = _l.pop(0)
            if isinstance(x, list):
                _l[0:0] = x
            else:
                out.append(x)
        return out


def find_null_indices(l):
    if type(l) is pd.Series:
        return l.index[l.isnull()]
    elif type(l) is list:
        return [idx for idx, x in enumerate(l) if (x is None) or (len(x) == 0)]
    else:
        raise('#TODO')


def remove_none(data, targets='all', axis=0, condition='any'):
    if type(data) in [pd.Series, pd.DataFrame]:
        if targets == 'all':
            targets = list(data.columns)
        if type(targets) != list:
            targets = [targets]
        indices = list()
        for target in targets:
            idx = data[target].isnull()
            indices.append(idx)
        indices = np.array(indices).T
        if axis == 0:
            if condition == 'any':
                idx_match = indices.any(axis=1)
            elif condition == 'all':
                idx_match = indices.all(axis=1)
            return data.loc[~idx_match,:]
        else:
            if condition == 'any':
                idx_match = indices.any(axis=0)
            elif condition == 'all':
                idx_match = indices.all(axis=0)
            return data.loc[:,~idx_match]
    else:
        return list(filter(None, data))


def remove_nan(data):
    return list(filter(np.nan, data))


def remove_inf(data):
    return list(filter(np.inf, data))


def remove_infeasible_values(data):
    return remove_nan(remove_inf(remove_none(data)))


class bcolors:
    magenta = '\033[95m'; m = '\033[95m'
    blue = '\033[94m'; b = '\033[94m'
    cyan = '\033[96m'; c = '\033[96m'
    green = '\033[92m'; g = '\033[92m'
    yellow = '\033[33m'; y = '\033[33m'
    red = '\033[91m'; r = '\033[91m'
    white = '\033[37m'; w = '\033[37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    end = '\033[0m'


def markrow(i):
    return '(Row ' + str(i) + ')'


def get_keys(dictionary, value):
    return [k for k, v in dictionary.items() if v == value]


def pplot(x, y):
    fig, ax = plt.subplots(1,1)
    plt.scatter(x, y)
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k-', zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.show()
    return


def dplot(x, bins=30):
    fig, ax = plt.subplots(1,1)
    ax.hist(x.reshape(-1, 1).squeeze(), bins=bins)
    plt.show()
    return


def trim_axes(axes, N):
    axes = axes.flat
    for ax in axes[N:]:
        ax.remove()
    return axes[:N]


def cprint(*args, color='cyan', inspect=True, newline=False):
    args = [str(arg) for arg in args]
    string = ' '.join(args)
    if inspect:
        parent_func_name = stack()[1][3]
        string = parent_func_name + ':: ' + string
    cstring = getattr(bcolors, color) + string + getattr(bcolors, 'end')
    print(cstring, flush=True)
    if newline or cstring == 'Done.':
        print('\n')
    return


def move_column_before(df, column_name, column_target):
    idx = int(np.where(np.array(df.columns) == column_target)[0][0])
    df.insert(idx, column_name, df.pop(column_name))
    return df


def move_column_after(df, column_name, column_target):
    idx = int(np.where(np.array(df.columns) == column_target)[0][0])
    df.insert(idx + 1, column_name, df.pop(column_name))
    return df


def checkexists(filedir, size_threshold=1):  # in byte
    if os.path.exists(filedir) and (os.path.getsize(filedir) >= size_threshold):
        return True
    else:
        return False


def check_df_size(df):
    df.info(memory_usage='deep')
    return


def optimize_df(df):
    cprint('Checking original data size...', color='c')
    check_df_size(df)
    for col in df.columns:
        data = df[col]
        dtype = data.dtype
        if any(data.isna() | data.isnull()):
            pass
        elif str(dtype) == 'int':
            c_min = data.min()
            c_max = data.max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                data = data.astype(np.int8)
            elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                data = data.astype(np.uint8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                data = data.astype(np.int16)
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                data = data.astype(np.uint16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                data = data.astype(np.int32)
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                data = data.astype(np.uint32)                    
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                data = data.astype(np.int64)
            elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                data = data.astype(np.uint64)
        elif str(dtype) in ['float16', 'float32', 'float64']:
            c_min = data.min()
            c_max = data.max()
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                data = data.astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                data = data.astype(np.float32)
            elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                data = data.astype(np.float64)
        elif str(dtype) == 'object':
            if isstringdata(data):
                if len(data) == len(data.unique()):
                    pass
                else:
                    data = data.astype('category')
                pass  # no room for optimization
            elif isnumericdata(data):
                _data = np.vstack(data)
                c_min = _data.min()
                c_max = _data.max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    _data = _data.astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    _data = _data.astype(np.float32)
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    _data = _data.astype(np.float64)
                data = np.split(_data, _data.shape[0])
            else:
                pass  # possibly 3rd party class instances
        else:
            raise('#TODO')
        df[col] = data
    cprint('Checking data size after optimization...', color='c')
    check_df_size(df)
    return df


def out_of_bounds(bounds, values):
    minval = bounds[0]
    maxval = bounds[1]
    if type(values) in [list, tuple] and len(values) == 2:  # mu and sigma are given
        mu = values[0]
        sigma = values[1]
        if type(mu) is np.ndarray:
            pairs = [(_mu, _sigma) for _mu, _sigma in zip(mu, sigma)]
            conditions = list(map(lambda pair: out_of_bounds(bounds, pair), pairs))
            within_bounds = ~np.array(conditions)
            if np.any(within_bounds):
                return False
            else:
                return True
        else:
            if (minval > mu + sigma) | (maxval < mu - sigma):
                return True
            else:
                return False
    elif isnumeric(values):
        value = values
        if (minval > value) | (maxval < value):
            return True
        else:
            return False
    else:
        raise('Unexpected behavior')


def replace_slash(string, reverse=False):
    if reverse:
        return string.replace('!%!', '/')
        return string.replace('!$!', '\\')
    else:
        return string.replace('/', '!%!')
        return string.replace('\\', '!$!')


class TimeoutError(Exception):
    pass


def wtimeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    from functools import wraps
    
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)
        
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print(e)
            finally:
                signal.alarm(0)
            return result
        return wraps(func)(wrapper)
    
    return decorator


def annotate_colname(x, **kws):
    ax = plt.gca()
    ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes, fontweight='bold')


if __name__ == '__main__':
    workingdir, rootdir, datadir, scriptdir, imgdir, srcdir = set_path(os.path.abspath(__file__))
    a = 1