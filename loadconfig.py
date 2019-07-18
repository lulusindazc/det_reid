#!/usr/bin/env python
# -*- coding:utf8 -*-
#
# @Time    : 18-10-28
# @Author  : Weibo Huang
# @Email   : weibohuang@pku.edu.cn
# @File    : loadconfig.py
# @Software: PyCharm
# @Description: 

import os
import yaml


class Configuration:
    def __init__(self, path='config.yaml'):
        #self._config_filepath = path
        self._f = open(path, 'r')
        cont = self._f.read()
        self.config = yaml.load(cont)


if __name__ == "__main__":
    this_dir, _ = os.path.split(__file__)
    print(this_dir)
    # config_filepath = os.path.join((this_dir), "config.yaml")
    config_filepath = os.path.join((this_dir), "config.yaml")
    print(config_filepath)
    f = open(config_filepath,'r', encoding='utf-8')
    cont = f.read()
    x = yaml.load(cont)
    print(type(x))
    #print(x)


    config = Configuration(path=config_filepath)
    #print(config.config['CameraConfig'])
    print(config.config['CameraConfig']['camera_7']['detect_region'])

