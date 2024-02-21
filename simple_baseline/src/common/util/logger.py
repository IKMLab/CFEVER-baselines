# -*- coding: utf-8 -*-
"""
Logger module
=============

This module provides functions for logging.


.. codeauthor:: Yi-Ting Li <yt.li.public@gmail.com>
"""
import logging

logger = logging.getLogger(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.DEBUG)
c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
logger.addHandler(c_handler)


# f_handler = logging.FileHandler('file.log')
# f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# f_handler.setLevel(logging.ERROR)
# f_handler.setFormatter(f_format)
# logger.addHandler(f_handler)
__all__ = ["logger"]
