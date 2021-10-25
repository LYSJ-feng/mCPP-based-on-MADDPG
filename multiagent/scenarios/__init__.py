import imp
import os.path as osp
# from multiagent.scenarios.custom_env_extend import Scenario


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)
