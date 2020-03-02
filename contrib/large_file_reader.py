#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    File Name: large_file_reader.py
    Description:
    
Created by Yong Bai on 2019/9/23 6:03 PM.
"""

import linecache


class CachedLineList:

    def __init__(self, fname):
        self._fname = fname

    def __getitem__(self, x):
        if isinstance(x, slice):
            start = x.start
            stop = x.stop
            step = 1 if x.step is None else x.step

            return [linecache.getline(self._fname, n+1)
                    for n in range(start, stop, step)]
        else:
            return linecache.getline(self._fname, x+1)

    def __getslice__(self, beg, end):
        # pass to __getitem__ which does extended slices also

        return self[beg:end:1]
