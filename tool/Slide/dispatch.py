#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys

# LRU算法直接返回对象 加快读取切片效率
from .cache import LRUCache
slides = LRUCache()


def openSlide(filename):
    ext = os.path.splitext(filename)[1][1:].lower()

    if filename in slides:
        return slides[filename]

    if ext == 'kfb':#宁波江丰
        from .KfbSlide.tool import KfbSlide
        #print(filename)
        slide = KfbSlide(filename)
    elif ext == 'sdpc':#深圳生强
        from .SdpcSlide.tool import SdpcSlide
        slide = SdpcSlide(filename)
    elif ext == 'mdsx':#麦克奥迪
        from .MdsxSlide.tool import MdsxSlide
        slide = MdsxSlide(filename)
    elif ext == 'hdx': #海德星5片机
        from .HdxSlide.tool import HdxSlide
        slide= HdxSlide(filename)
    #志盈和优纳linux版本还有些问题
    elif ext == 'zyp' and sys.platform=='win32': #志盈
        from .ZYPSlide.tool import ZYPSlide
        slide = ZYPSlide(filename)
    elif ext == 'tmap': #优纳
        from .TmapSlide.tool import TmapSlide
        slide = TmapSlide(filename)
    else:  #openslide支持的切片格式
        from .OtherSlide.tool import OtherSlide
        slide = OtherSlide(filename)

    slides[filename] = slide

    return slide

