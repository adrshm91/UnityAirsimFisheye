import OpenEXR
import PIL.Image as Image
import Imath
import numpy as np

file_name = './0000000_depth.exr'

pt = Imath.PixelType(Imath.PixelType.FLOAT)
img_exr = OpenEXR.InputFile(file_name)
dw = img_exr.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
redstr = img_exr.channel('R', pt)
red = np.fromstring(redstr, dtype = np.float32)
red.shape = (size[1], size[0])
rgbf = [Image.frombytes("F", size, img_exr.channel(c, pt)) for c in "RGB"]
extrema = [im.getextrema() for im in rgbf]
darkest = min([lo for (lo, hi) in extrema])
lighest = max([hi for (lo, hi) in extrema])
scale = 255 / (lighest - darkest)
def normalize_0_255(v):
    return (v * scale) + darkest
rgb8 = [im.point(normalize_0_255).convert("L") for im in rgbf]
Image.merge("RGB", rgb8).save('./0000000_depth.jpg')