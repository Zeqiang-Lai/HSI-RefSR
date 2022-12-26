import collections
from functools import reduce
import ast
import argparse
from pathlib import Path
import json
import yaml
import sys
import os
import shutil

from munch import Munch
from colorama import init, Fore
init(autoreset=True)

from ._util import action_confirm

def basic_args(description=''):
    """
    fresh new training:
        python run.py train -c [config.yaml] -s [save_dir]
    resume training:
        python run.py train -s [save_dir] -r latest
    resume training with overrided config
        python run.py train -c [override.yaml] -s [save_dir] -r latest  # this will override the original config
        python run.py train -c [config.yaml] -s [save_dir] -r latest -n [new_save_dir]
    test
        python run.py test -s [save_dir] -r best
    test with overrided config
        python run.py test -s [save_dir] -r best -c config.yaml # test won't override the original config, but save the override config in test directory
        python run.py test -s [save_dir] -r best -n new_save_dir # save in a new place
    """

    args = argparse.ArgumentParser(description=description)
    args.add_argument('mode', type=str, help='running mode',
                      choices=['train', 'test', 'debug'])
    args.add_argument('-c', '--config', nargs='*', default=None, type=str,
                      help='config file(s) path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='resume to # checkpoint (default: None), e.g. best | latest | epoch# or a complete path')
    args.add_argument('-d', '--device', default='cuda', type=str,
                      help='indices of GPUs to enable (default: cuda)')
    args.add_argument('-s', '--save_dir', default='saved', type=str, required=True,
                      help='path of log directory (default: saved)')
    args.add_argument('-n', '--new_save_dir', default=None, type=str,
                      help='path of new log directory (default: new_saved)')
    args.add_argument('--override', default=None, type=str,
                      help='custom value to override the corrsponding key in config file.'
                      'format: "key.key=value; key2=value2", each KV pair must be seperate by a semicolon.')
    args.add_argument('--reset', action='store_true', default=False, 
                      help='remove the old log dir if exists, only used in train mode')
    args = args.parse_args()

    vars(args)['resume_dir'] = args.save_dir
    if args.new_save_dir is not None:
        args.save_dir = args.new_save_dir

    if args.mode == 'train' and args.reset:
        if os.path.exists(args.save_dir) and \
            action_confirm(Fore.RED + f'Do you really want to reset the old log?\nPath=({args.save_dir})'):
            shutil.rmtree(args.save_dir)
    
    if args.mode == 'test':
        assert args.resume is not None, 'resume cannot be None in test mode'

    if args.resume:
        if args.config and args.new_save_dir:
            cfg = read_config(args.config)
        else:
            resume_config_path = Path(args.resume_dir) / 'config.yaml'
            cfg = read_yaml(resume_config_path)
            if args.config:
                 deep_update(cfg, read_config(args.config))
    else:
        if args.config is None:
            print(Fore.RED + 'Warning: default config not founded, forgot to specify a configuration file?')
            cfg = {'engine': {}}
        else:
            cfg = read_config(args.config)

    os.makedirs(os.path.join(args.save_dir, 'history'), exist_ok=True)
    with open(os.path.join(args.save_dir, 'history', 'cmd.log'), 'a') as f:
        cmd = 'python ' + ' '.join(sys.argv) + '\n'
        f.write(cmd)

    _set_custom_args(cfg, args.override)

    return args, Munch.fromDict(cfg)


def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def read_config(configs):
    cfg = {}
    # the first must be full path, can omit extension, 
    # the subseqeunt can use file name only
    base_dir = os.path.dirname(configs[0])
    for config in configs:
        # if without extension, append it
        if not config.endswith('.yaml'):
            config = config + '.yaml'
        # if file not exists, try add base_dir
        if not os.path.exists(config):
            config = os.path.join(base_dir, config)
        deep_update(cfg, read_yaml(config))
    return cfg

# ---------------------------------------------------------------------------- #
#                          Parse custom overried args                          #
# ---------------------------------------------------------------------------- #

def get_default(a:dict, b):
    return a.get(b, {})

def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    return reduce(get_default, items, root)


def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(root, items[:-1])[items[-1]] = value


def _eval(val):
    try:
        val = ast.literal_eval(val)
    except:
        return val
    return val


def _remove_whitespace(array):
    return [s.strip() for s in array]


def _set_custom_args(origin: dict, override: str):
    """ --override "module.lr=0.01; engine.max_epochs=10"
    """
    if override is None:
        return
    options = _remove_whitespace(override.split(';'))
    for option in options:
        keys, value = tuple(_remove_whitespace(option.split('=')))
        keys = _remove_whitespace(keys.split('.'))
        value = _eval(value)
        set_by_path(origin, keys, value)

# ---------------------------------------------------------------------------- #
#                            Read/Write config file                            #
# ---------------------------------------------------------------------------- #


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.safe_dump(content, handle, indent=4, sort_keys=False)
