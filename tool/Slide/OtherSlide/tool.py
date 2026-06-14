#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os.path,math,sys,io
if sys.platform=='win32':
    dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PATH"] = os.path.join(dir,'openslidec')+ ';'+ os.environ["PATH"]
    os.add_dll_directory(os.path.join(dir,'openslidec'))
    from . import openslide
    from .openslide.deepzoom import DeepZoomGenerator
else:
    import openslide
    from openslide.deepzoom import DeepZoomGenerator
from PIL import Image,ImageCms
import tifffile
import cv2
import numpy as np
import json
from ..SlideBase import SlideBase


Image.MAX_IMAGE_PIXELS = None

class OtherSlide(SlideBase):
    def __init__(self,filename):
        # print("OpenSlide")
        self.filename = filename
        self.icc2rgb = None
        # If suffix is bif or tif (Roche DP200 format)
        if os.path.splitext(self.filename)[1].lower() in ['.tif','.bif','.svs']:
            slide_fp = tifffile.TiffFile(filename)
            try:
                try:
                    icc_bytes = slide_fp.pages[2].tags[34675].value
                except:
                    icc_bytes = slide_fp.pages[0].tags[34675].value
                icc_prf = io.BytesIO(icc_bytes)
                rgbp = ImageCms.createProfile('sRGB')
                self.icc2rgb = ImageCms.buildTransformFromOpenProfiles(icc_prf,rgbp,'RGB','RGB')
            except Exception as e:
                pass
                
        if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', 'png', '.bmp']:
            try:
                self.slide = openslide.ImageSlide(filename)
            except:
                self.slide=  openslide.OpenSlide(filename)
        else:
            self.slide = openslide.open_slide(filename)
        self.width, self.height = self.slide.dimensions
        SlideBase.__init__(self)
        self.getTile( 0,0 , self.maxlvl )#初始化完成后，才释放lock

    def read(self, location=[0,0], size=None, scale=1.0, greyscale=False):
        '''
        :param location: (x, y) at level=0
        :param size: (width, height)
        :param scale: resize scale, scale>1 -> zoom out, scale<1 -> zoom in
        :param greyscale: if True, convert image to greyscale
        :return: a numpy image,  np_img.shape=[height, width, channel=1 or 3]
        '''
        if size == None:
            width, height = self.width, self.height
        else:
            width, height = size

        crop_start_x, crop_start_y = location
        crop_level = self.slide.get_best_level_for_downsample(scale)
        resize_ratio = self.slide.level_downsamples[crop_level]/scale

        # make sure the crop region is inside the slide
        crop_start_x = math.ceil(min(max(crop_start_x, 0), self.width))
        crop_start_y = math.ceil(min(max(crop_start_y, 0), self.height))
        crop_end_x = math.ceil(min(max(width+crop_start_x, 0), self.width))
        crop_end_y = math.ceil(min(max(height+crop_start_y, 0), self.height))

        crop_width = math.ceil((crop_end_x - crop_start_x)/self.slide.level_downsamples[crop_level])
        crop_height = math.ceil((crop_end_y - crop_start_y)/self.slide.level_downsamples[crop_level])

        if crop_height == 0 or crop_width == 0:
            return None

        crop_region = self.slide.read_region(
            location=(crop_start_x, crop_start_y),
            level=crop_level,
            size=(crop_width, crop_height)
        )

        if self.icc2rgb:
            crop_region = ImageCms.applyTransform(crop_region,self.icc2rgb)

        crop_region = np.array(crop_region)[:, :, 0:3]

        if greyscale:
            crop_region = 0.2989*crop_region[:,:,0] + 0.5870*crop_region[:,:,1] + 0.1140*crop_region[:,:,2]
            crop_region = crop_region[:,:,np.newaxis]

        crop_region = cv2.resize(crop_region, (math.ceil(crop_width*resize_ratio), math.ceil(crop_height*resize_ratio)))
        return crop_region


    def getTile(self,x,y,z):
        dzi_obj = DeepZoomGenerator(self.slide, tile_size=1024, overlap=0)
        # print(x,y,z)
        tile = dzi_obj.get_tile(level=z, address=(x,y))
        if self.icc2rgb:
            tile = ImageCms.applyTransform(tile,self.icc2rgb)
        return tile


    def saveLabel(self,path):
        try:
            self.slide.associated_images["label"].save(path)
        except:
            try:
                self.slide.associated_images['macro'].save(path)
            except:
                pass


    @property
    def mpp(self):
        mpp = None
        try:
            slide_properties = self.slide.properties
            if 'openslide.mpp-x' in slide_properties:
                mpp = float(slide_properties['openslide.mpp-x'])
                return mpp
            if 'tiff.XResolution' in slide_properties:
                unit = slide_properties['tiff.ResolutionUnit']
                if unit == 'centimeter':
                    return 10000 / float(slide_properties['tiff.XResolution'])
                if unit == 'inch':
                    return 25400 / float(slide_properties['tiff.XResolution'])
        except:
            pass

        return mpp

