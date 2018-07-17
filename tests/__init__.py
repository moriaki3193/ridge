# -*- coding: utf-8 -*-
from os.path import (
    abspath as _abspath,
    dirname as _dirname,
    join as _join)

BASEDIR = _dirname(_dirname(_abspath(__file__)))
DATADIR = _join(BASEDIR, 'tests', 'data')
