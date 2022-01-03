#!/usr/bin/env python3

from inspect import stack
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import transforms
from matplotlib.backend_bases import MouseButton
from pprint import pprint
import numpy as np
from numbers import Number
import random
from functools import reduce
import copy

def xfrm_to_deg(m):
    # https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix
    a,c,b,d,tx,ty = m.to_values()
    return np.degrees(np.arctan2(-b,a))

def matrix_to_pos(m):
    # https://math.stackexchange.com/questions/13150/extracting-rotation-scale-values-from-2d-transformation-matrix
    a,c,b,d,tx,ty = m.to_values()
    return xy(tx, ty)

# A package consists of 
# * A body outline
# * An array of pins and their centers
# * A bounding box

class xy:
    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], Number) and isinstance(args[1], Number):
            self.x = args[0]
            self.y = args[1]
        elif args and isinstance(args[0], xy):
            self.x = args[0].x
            self.y = args[0].y
        elif args and isinstance(args[0], wh):
            self.x = args[0].w
            self.y = args[0].h
        elif args and isinstance(args[0], tuple):
            self.x = args[0][0]
            self.y = args[0][1]
        elif 'x' in kwargs and 'y' in kwargs:
            self.x = kwargs['x']
            self.y = kwargs['y']
    def __getitem__(self, i):
        if(i == 0):
            return self.x
        if(i == 1):
            return self.y
        raise IndexError
    def __setitem__(self, i, val):
        if(i == 0):
            self.x = val
        elif(i == 1):
            self.y = val
        else:
            raise IndexError
    def __iter__(self): 
        return iter((self.x, self.y))
    def __add__(self, *args):
        if isinstance(args[0], xy):
            return xy(self.x + args[0].x, self.y + args[0].y)
        elif isinstance(args[0], Number):
            return xy(self.x + args[0], self.y + args[0])
    def __sub__(self, *args):
        if isinstance(args[0], xy):
            return xy(self.x - args[0].x, self.y - args[0].y)
        elif isinstance(args[0], Number):
            return xy(self.x - args[0], self.y - args[0])
    def __mul__(self, *args):
        if isinstance(args[0], Number):
            return xy(self.x * args[0], self.y * args[0])
    def __truediv__(self, *args):
        if isinstance(args[0], Number):
            return xy(self.x / args[0], self.y / args[0])
    def __repr__(self):
        return f'xy({self.x},{self.y})'
    def swap(self):
        tmp = self.x
        self.x = self.y
        self.y = tmp
    def swap_new(self):
        return xy(self.y, self.x)
    def mag(self):
        return np.sqrt(self.x**2 + self.y**2)
    def magsquared(self):
        return self.x**2 + self.y**2
    def norm(self):
        m = self.mag()
        return xy(self.x / m, self.y / m)

class wh:
    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], Number) and isinstance(args[1], Number):
            self.w = args[0]
            self.h = args[1]
        elif args and isinstance(args[0], wh):
            self.w = args[0].w
            self.h = args[0].h
        elif args and isinstance(args[0], xy):
            self.w = args[0].x
            self.h = args[0].y
        elif args and isinstance(args[0], tuple):
            self.w = args[0][0]
            self.h = args[0][1]
        elif 'width' in kwargs and 'height' in kwargs:
            self.w = kwargs['width']
            self.h = kwargs['height']
    def __getitem__(self, i):
        if(i == 0):
            return self.w
        if(i == 1):
            return self.h
        raise IndexError
    def __setitem__(self, i, val):
        if(i == 0):
            self.w = val
        elif(i == 1):
            self.h = val
        else:
            raise IndexError
    def __iter__(self): 
        return iter((self.w, self.h))
    def __mul__(self, *args):
        return wh(self.w * args[0], self.h * args[0])
    def __truediv__(self, *args):
        return wh(self.w / args[0], self.h / args[0])
    def __repr__(self):
        return f'wh({self.w},{self.h})'
    def swap(self):
        tmp = self.w
        self.w = self.h
        self.h = tmp
    def swap_new(self):
        return wh(self.h, self.w)

class Shape:
    def __init__(self):
        self.transform = transforms.Affine2D()
        self.children = []
    def translate(self, val: xy):
        # do something with matrix
        self.transform.translate(val.x, val.y)
        self._compute_points()
    def translate_to(self, val: xy):
        # do something with matrix
        a,b,c,d,e,f = self.transform.to_values()
        self.transform = transforms.Affine2D.from_values(a,b,c,d,*val)
        self._compute_points()
    def rotate(self, val: float):
        # Rotate by degrees
        # Todo: add optional argument for xy position to rotate around
        self.transform.rotate_deg(val)
        self._compute_points()
    def rotate_to(self, val: float):
        # Undo the previous rotation
        oldrot = xfrm_to_deg(self.transform)
        self.transform.rotate_deg(-oldrot)
        self.transform.rotate_deg(val)
        self._compute_points()
    
class Rectangle(Shape):
    """A rectangle is not necessarily centered about its origin, nor does it 
       necessarily have its lower left corner at its origin, so it needs both
       xy position and wh dimensions.  In most cases, xy will be zero, or
       will be computed if 'centered' is given as an option."""
    def __init__(self, *args, **kwargs):
        Shape.__init__(self)
        if args and isinstance(args[0], xy) and isinstance(args[1], wh):
            # given xy and wh directly
            self._xy = copy.copy(args[0])
            self._wh = copy.copy(args[1])
        if args and isinstance(args[0], wh):
            # given only wh directly
            if 'centered' in kwargs and kwargs['centered']:
                self._xy = xy(args[0]) / (-2.0)
            else:
                self._xy = xy(0.0, 0.0)
            self._wh = args[0]
        elif args                        and \
             len(args) == 4              and \
             isinstance(args[0], Number) and \
             isinstance(args[1], Number) and \
             isinstance(args[2], Number) and \
             isinstance(args[3], Number):
             # given four numbers directly
             self._xy = xy(args[0], args[1])
             self._wh = wh(args[2], args[3])
        elif args                        and \
            len(args) == 2              and \
            isinstance(args[0], Number) and \
            isinstance(args[1], Number):
            # given two numbers directly
            if 'centered' in kwargs and kwargs['centered']:
                self._xy = xy(-args[0]/2, -args[1]/2)
            else:
                self._xy = xy(0.0, 0.0)
            self._wh = wh(args[0], args[1])
        elif args and isinstance(args[0], Rectangle):
            # given another rectangle directly
            self._xy = copy.copy(args[0].xy)
            self._wh = copy.copy(args[0].wh)
            self.transform = args[0].transform.frozen()
        elif 'xy' in kwargs and 'wh' in kwargs:
            # given pos and wh in dict
            self._xy = xy(*kwargs['xy'])
            self._wh = wh(*kwargs['wh'])
        elif 'wh' in kwargs:
            # given only wh in dict
            if 'centered' in kwargs and kwargs['centered']:
                self._xy = xy(*kwargs['wh']) / (-2.0)
            else:
                self._xy = xy(0.0, 0.0)
            self._wh = wh(*kwargs['wh'])
        elif 'x' in kwargs and 'y' in kwargs and 'width' in kwargs and 'height' in kwargs:
            # given four values in dict
            self._xy = xy(kwargs['x'], kwargs['y'])
            self._wh = wh(kwargs['width'], kwargs['height'])
        if 'patchargs' in kwargs:
            self.patchargs = kwargs['patchargs']
        else:
            self.patchargs = {}
        assert hasattr(self, '_xy')
        assert hasattr(self, '_wh')
        self._compute_points()
    def _compute_points(self):       
        self.matrix = np.matrix(self.corners).transpose()
        self.matrix = np.insert(self.matrix, [2], 1, axis=0)
        self.matrix = self.transform.get_matrix() * self.matrix
    def draw(self, ax: plt.Axes, xfrm=transforms.Affine2D()):
        # xfrm is parent's position and rotation
        pts = xfrm * self.matrix
        pts = pts[0:2, :].transpose() # throw away bottom row and get N x 2
        patch = patches.Polygon(pts, **self.patchargs)
        ax.add_patch(patch)
    def __getitem__(self, i):
        if (i == 0):
            return self._xy.x
        if (i == 1):
            return self._xy.y
        if (i == 2):
            return self._wh.w
        if (i == 3):
            return self._wh.h
        raise IndexError
    def __setitem__(self, i, val):
        if (i == 0):
            self._xy.x = val
        elif (i == 1):
            self._xy.y = val
        elif (i == 2):
            self._wh.w = val
        elif (i == 3):
            self._wh.h = val
        else:
            raise IndexError
    def __repr__(self):
        return f'Rectangle({self.x},{self.y},{self.width},{self.height},{self.transform})'
    @property
    def xy(self):
        return self._xy
    @xy.setter
    def xy(self, val):
        self._xy = val
        self._compute_points()
    @property
    def wh(self):
        return self._wh
    @wh.setter
    def wh(self, val):
        self._wh = val
        self._compute_points()
    @property
    def width(self):
        return self._wh.w
    @width.setter
    def width(self, val):
        self._wh.w = val
        self._compute_points()
    @property
    def height(self):
        return self._wh.h
    @height.setter
    def height(self, val):
        self._wh.h = val
        self._compute_points()
    @property
    def x(self):
        return self._xy.x
    @x.setter
    def x(self, val):
        self._xy.x = val
        self._compute_points()
    @property
    def y(self):
        return self._xy.y
    @y.setter
    def y(self, val):
        self._xy.y = val
        self._compute_points()
    @property
    def left(self):
        # Return minimum x value after transform
        pts = self.points
        pts = pts[:,0]
        return np.min(pts)
    @property
    def right(self):
        # Return maximum x value after transform
        pts = self.points
        pts = pts[:,0]
        return np.max(pts)
    @property
    def bottom(self):
        # Return minimum y value after transform
        pts = self.points
        pts = pts[:,1]
        return np.min(pts)
    @property
    def top(self):
        # Return maximum y value after transform
        pts = self.points
        pts = pts[:,1]
        return np.max(pts)
    @property
    def points(self):
        # Returns Nx2 array of points after transform
        # Assumes self.matrix is 3xN
        # where bottom row is all ones, and N is total number of points
        arr = np.asarray(self.matrix) # convert to matrix
        arr = arr[0:2, :].transpose() # throw away bottom row and get N x 2
        return arr
    @property
    def corners(self):
        # Returns pairs of points before transform
        return ((self.x,              self.y),
                (self.x + self.width, self.y),
                (self.x + self.width, self.y + self.height),
                (self.x,              self.y + self.height))
    def area(self):
        return self._wh.w * self._wh.h

class RectPin:
    def __init__(self, dims=wh(1.0,1.0), pos=xy(0.0, 0.0), name='pin'):
        self.rect = Rectangle(wh=wh(dims), centered=True, patchargs={'color':'red', 'alpha':0.5})
        self.rect.translate(xy(pos))
        self.name = name
        self.bbox = Rectangle(self.rect)
    def __repr__(self):
        return f'RectPin({self.rect.width}, {self.rect.height}, {self.rect.transform.get_matrix()})'
    def translate(self, val: xy):
        self.rect.translate(val)
        self.bbox.translate(val)
    def rotate(self, val: float):
        self.rect.rotate(val)
        self.bbox.rotate(val)
    def draw(self, ax, xfrm=transforms.Affine2D()):
        # xfrm is parent's translation and rotation
        self.rect.draw(ax, xfrm)
        localxfrm = self.rect.transform + xfrm
        pos = localxfrm.to_values()[-2:]
        rot = xfrm_to_deg(localxfrm)
        #ax.text(*pos, s=self.name, va='center', ha='center', rotation=rot, clip_on=True)

class Circle(Shape):
    """A Circle is by definition centered around its own origin, 
       with only diameter and colors, etc., as its properties,
       so no xy position needed."""
    def __init__(self, dia, *args, **kwargs):
        Shape.__init__(self)
        self.dia = dia
        self.matrix = np.array([[0],[0],[1]])
        if 'patchargs' in kwargs:
            self.patchargs = kwargs['patchargs']
        else:
            self.patchargs = {}
    @property
    def r(self):
        return self.dia / 2.0
    @r.setter
    def r(self, val):
        self.dia = val * 2.0
    def _compute_points(self):       
        self.matrix = np.matrix(self.corners).transpose()
        self.matrix = np.insert(self.matrix, [2], 1, axis=0)
        self.matrix = self.transform.get_matrix() * self.matrix
    def draw(self, ax: plt.Axes, xfrm=transforms.Affine2D()):
        # xfrm is parent's position and rotation
        pts = xfrm * self.matrix
        pts = pts[0:2, :].transpose() # throw away bottom row and get N x 2
        ctr = tuple(np.asarray(pts)[0])
        patch = patches.Circle(ctr, self.r, fill=True, **self.patchargs)
        ax.add_patch(patch)
    def __repr__(self):
        return f'Circ({self.dia})'
    @property
    def points(self):
        # Returns Nx2 array of points after transform
        # Assumes self.matrix is 3xN
        # where bottom row is all ones, and N is total number of points
        arr = np.asarray(self.matrix) # convert to matrix
        arr = arr[0:2, :].transpose() # throw away bottom row and get N x 2
        return arr
    @property
    def corners(self):
        # Returns pairs of points before transform
        return ((0.0, 0.0),)


# Todo: clean up constructor interface so it's consistent 
# Todo: with Rectangle, ie with copy, different permutations of args, etc.
class CircPin:
    def __init__(self, dia: Number, pos=xy(0.0, 0.0), name='pin'):
        self.circ = Circle(dia, patchargs={'color':'red', 'alpha':0.5})
        self.circ.translate(xy(pos))
        self.name = name
        self.bbox = Rectangle(pos.x-self.circ.r, pos.y-self.circ.r, dia, dia)
    def __repr__(self):
        return f'CircPin({self.dia})'
    def translate(self, val: xy):
        # Move the circle
        self.circ.translate(val)
        self.bbox.translate(val)
    def rotate(self, val: xy):
        # Yes a circle can rotate if it's not at the origin
        self.circ.rotate(val)
        self.bbox.rotate(val)
    def draw(self, ax: plt.Axes, xfrm=transforms.Affine2D()):
        # xfrm is parent's translation and rotation
        self.circ.draw(ax, xfrm)
        localxfrm = self.circ.transform + xfrm
        pos = localxfrm.to_values()[-2:]
        rot = xfrm_to_deg(localxfrm)
        #ax.text(*pos, s=self.name, va='center', ha='center', rotation=rot, clip_on=True)

packspecs = {
    'RCMF01005': {'W': 0.4,   'H': 0.2,   'pintype': 'two-pin-passive', 'pindims': {'inner': 0.2,         'outer': 0.5,        'minordim': 0.2}},
    'RCMF0201':  {'W': 0.6,   'H': 0.3,   'pintype': 'two-pin-passive', 'pindims': {'inner': 0.3,         'outer': 1.0,        'minordim': 0.4}},
    'RCMF0402':  {'W': 1.0,   'H': 0.5,   'pintype': 'two-pin-passive', 'pindims': {'inner': 0.5,         'outer': 1.5,        'minordim': 0.6}},
    'RCMF0603':  {'W': 1.55,  'H': 0.8,   'pintype': 'two-pin-passive', 'pindims': {'inner': 0.8,         'outer': 2.1,        'minordim': 0.9}},
    'RCMF0805':  {'W': 2.0,   'H': 1.25,  'pintype': 'two-pin-passive', 'pindims': {'inner': 1.2,         'outer': 3.0,        'minordim': 1.3}},
    'RCMF1206':  {'W': 3.2,   'H': 1.6,   'pintype': 'two-pin-passive', 'pindims': {'inner': 2.2,         'outer': 4.2,        'minordim': 1.6}},
    'RCMF1210':  {'W': 3.2,   'H': 2.5,   'pintype': 'two-pin-passive', 'pindims': {'inner': 2.2,         'outer': 4.2,        'minordim': 2.8}},
    'RCMF2010':  {'W': 5.0,   'H': 2.5,   'pintype': 'two-pin-passive', 'pindims': {'inner': 3.5,         'outer': 6.1,        'minordim': 2.8}},
    'RCMF2512':  {'W': 6.3,   'H': 3.2,   'pintype': 'two-pin-passive', 'pindims': {'inner': 4.9,         'outer': 8.0,        'minordim': 3.5}},
    'LT4312f' :  {'W': 3.0,   'H': 4.0392,'pintype': 'dual-row-ic',     'pindims': {'inner': 3.2,         'outer': 5.23,       'minordim': 0.305,       'numpins': 16,     'minorpitch': 0.5        }},
    'SX9376'  :  {'W': 1.8,   'H': 1.9,   'pintype': 'quad-row-ic',     'pindims': {'inner': (1.1, 1.0),  'outer': (2.3, 2.2), 'minordim': (0.2, 0.2),  'numpins':  [6,6], 'minorpitch': (0.4, 0.4) }},
    'LPS22DF' :  {'W': 2.0,   'H': 2.0,   'pintype': 'quad-row-ic',     'pindims': {'inner': (1.25,1.25), 'outer': (2.3,2.3),  'minordim': (0.25,0.25), 'numpins':  [6,4], 'minorpitch': (0.5,0.5)  }},
    'SLG51001':  {'W': 1.675, 'H': 1.675, 'pintype': 'ball-array',      'pindims': {'rows': 4, 'cols': 4, 'pitch': 0.4, 'dia': 0.27}}
    }

class Package:
    def __init__(self, name, pos=xy(0.0, 0.0), rot=0.0): #name, body, pins, bbox):
        self.name = name
        self.pos = xy(0.0, 0.0)
        self.rot = 0.0 # <-- these update in the initial rotate and translate below
        pdict = packspecs[name]
        majordim = pdict['W']
        minordim = pdict['H']
        pindims = pdict['pindims']
        self.body = Rectangle(majordim, minordim, centered=True, patchargs={'color':'blue', 'alpha':0.5})
        self.pins = []
        ptype = pdict['pintype']
        pindims = pdict['pindims']
        if ptype == 'two-pin-passive':
            self.pins.extend(make_dual_pins(**pindims))
        elif ptype == 'dual-row-ic':
            self.pins.extend(make_dual_pins(**pindims))
        elif ptype == 'dual-row-connector':
            pass
        elif ptype == 'quad-row-ic':
            self.pins.extend(make_quad_pins(**pindims))
        elif ptype == 'ball-array':
            self.pins.extend(make_regular_ball_array(**pindims))
        self.bbox = make_bounding_box(self.body, self.pins)
        self.rotate(rot)
        self.translate(pos)
    def __repr__(self):
        return f'Package({self.name},{self.pos})'
    def translate(self, val: xy):
        self.pos = self.pos + val
        self.body.translate(val)
        self.bbox.translate(val)
    def translate_to(self, val: xy):
        self.pos = val
        self.body.translate_to(val)
        self.bbox.translate_to(val)
    def rotate(self, val: float):
        self.rot = self.rot + val
        self.body.rotate(val)
        self.bbox.rotate(val)
    def rotate_to(self, val: float):
        self.rot = self.rot
        self.body.rotate_to(val)
        self.bbox.rotate_to(val)
    def draw(self, ax: plt.Axes, xfrm=transforms.Affine2D()):
        # Generally for packages, the built-in transform is contained
        # in the body (self.body.transform) and is the "true" position 
        # of the package, however if this is ever used within another 
        # body, then the xfrm argument would receive the transform of 
        # the holding body.

        # Extract rotation from transform, draw body and package type
        self.body.draw(ax, xfrm)
        localxfrm = self.body.transform + xfrm
        #pos = localxfrm.to_values()[-2:]
        #rot = xfrm_to_deg(localxfrm)
        #ax.text(*pos, s=self.name, va='center', ha='center', rotation=rot, clip_on=True)
        # TODO: Add refdes

        # Draw pins and bbox, taking add'l xform from package
        for pin in self.pins:
            pin.draw(ax, self.body.transform)
        self.bbox.draw(ax)

def pitch_to_offset(pitch, numpins):
    """Given pitch and number of pins, return
       offset needed to center the pins"""
    return -(pitch * (0.5 * numpins - 0.5))

def make_dual_pins(inner, outer, minordim, numpins=2, minorpitch=0.0):
    """ Returns numpins pins, computing centers and widths.
        * inner and outer are pad edge distances along the orientation axis.
        * minordim is height of pad
        * numpins is the number of pins.  Must be an even integer.
        * minorpitch is the ctr-ctr distance between pads along minor axis,
          ignored if numpins <= 2
        Errors result if: any value is negative, numpins < 2 or odd, etc.
        """

    assert(numpins >= 2)
    assert((numpins % 2) == 0)
    assert(inner > 0)
    assert(outer > inner)

    majordim = (outer - inner) / 2.0      # width of pad along main orientation
    majorpitch = inner + majordim         # ctr-ctr distance between pads along main orientation
    pins = []
    
    dim = wh(majordim, minordim)
    for i in range(1, numpins+1):
        left = i<=numpins/2 # heading down or up
        vi = i if left else numpins-i+1 # virtual-i
        majorpos = -majorpitch/2.0 if left else majorpitch/2.0
        minorpos = minorpitch * (numpins/4 - vi + 0.5)
        pos = xy(majorpos, minorpos)
        pins.append(RectPin(dim, pos, name=f'{i}'))
    return pins

def make_quad_pins(inner, outer, minordim, numpins, minorpitch):
    """ Same as make_dual_pins, except expects each argument to be a 2-tuple or 2-list
    Assume same number of pins left and right, and same number of pin top and bottom,
    so numpins = [n1, n2], where n1 is total pins shared between left and right, and
    n2 is total pins shared between top and bottom.  So n1 and n2 must be 0 or an even
    number.  This function becomes the equivalent of make_dual_pins if the top and bottom
    numpins are zero. """

    assert(len(inner) == 2)
    assert(len(outer) == 2)
    assert(len(minordim) == 2)
    assert(len(numpins) == 2)
    assert(len(minorpitch) == 2)
    assert(outer[0] > inner[0])
    assert(outer[1] > inner[1])
    assert(numpins[0] % 2 == 0)
    assert(numpins[1] % 2 == 0)

    # Four loops in pin order
    lrqty = numpins[0]//2
    tbqty = numpins[1]//2
    pins = []
    left   = range(1, lrqty+1)
    bottom = range(lrqty+1, lrqty+tbqty+1)
    right  = range(lrqty+tbqty+1, numpins[0]+tbqty+1)
    top    = range(numpins[0]+tbqty+1, numpins[0]+numpins[1]+1)

    majordim = (outer[0] - inner[0]) / 2.0      # width of pad along main orientation
    majorpitch = inner[0] + majordim            # ctr-ctr distance between pads along main orientation
    dim = wh(majordim, minordim[0])
    for i in left:
        majorpos = -majorpitch/2.0
        minorpos = minorpitch[0] * (lrqty/2 - i + 0.5)
        pos = xy(majorpos, minorpos)
        pins.append(RectPin(dim, pos, name=f'{i}'))

    for i in right:
        vi = numpins[0] + tbqty - i + 1 # virtual-i
        majorpos = majorpitch/2.0
        minorpos = minorpitch[0] * (lrqty/2 - vi + 0.5)
        pos = xy(majorpos, minorpos)
        pins.append(RectPin(dim, pos, name=f'{i}'))

    majordim = (outer[1] - inner[1]) / 2.0      # width of pad along main orientation
    majorpitch = inner[1] + majordim            # ctr-ctr distance between pads along main orientation
    dim = wh(minordim[1], majordim)
    for i in bottom:
        vi = i - lrqty
        majorpos = -majorpitch/2.0
        minorpos = minorpitch[1] * (vi - tbqty/2.0 - 0.5 )
        pos = xy(minorpos, majorpos)
        pins.append(RectPin(dim, pos, name=f'{i}'))

    for i in top:
        vi = numpins[0] + numpins[1] - i + 1 # virtual-i
        majorpos = majorpitch/2.0
        minorpos = minorpitch[1] * (vi - tbqty/2.0 - 0.5)
        pos = xy(minorpos, majorpos)
        pins.append(RectPin(dim, pos, name=f'{i}'))

    return  pins

def num_to_letter(n, minpos=1):
    """convert positive decimal integer n to equivalent in another base (2-36).
    From https://stackoverflow.com/questions/2267362/how-to-convert-an-integer-to-a-string-in-any-base
    """

    digits = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    base = 26
    s = ""
    while 1:
        r = n % base
        s = digits[r] + s
        n = n // base
        if n == 0:
            break
    if len(s) < minpos:
        s = 'A' * (minpos - len(s)) + s
    return s

def make_regular_ball_array(rows, cols, pitch, dia):
    import math
    posx = pitch_to_offset(pitch, cols)
    posy = pitch_to_offset(pitch, rows)
    num_letters = int(math.log(rows, 26)) + 1
    num_digits  = int(math.log(cols, 10)) + 1
    pinout = []
    for row in range(rows):
        for col in range(cols):
            name = f'{num_to_letter(row, num_letters)}{col:0{num_digits}d}'
            pos = xy(col * pitch + posx, -(row * pitch + posy))
            pin = CircPin(dia, pos=pos, name=name)
            pinout.append(pin)
    return pinout

def union_box(box1: Rectangle, box2: Rectangle) -> Rectangle:
    """Returns the rectangle that is the bounding box of the two given rectangles.
    This doesn't take into consideration any rotation that may be applied to the rectangles"""
    left  = min(box1.left,   box2.left)
    right = max(box1.right,  box2.right)
    bot   = min(box1.bottom, box2.bottom)
    top   = max(box1.top,    box2.top)
    return Rectangle(left, bot, right - left, top - bot, patchargs={'color':'green', 'alpha':0.5, 'fill':None})

def make_bounding_box(body: Rectangle, pins: list) -> Rectangle:
    """Bounding box relative to body coordinates"""
    rlist = (body, *(p.bbox for p in pins))
    return reduce(union_box, rlist)

def make_packages_bbox(packages: list) -> Rectangle:
    rlist = (p.bbox for p in packages)
    return reduce(union_box, rlist)

def bbox_intersect(r1: Rectangle, r2: Rectangle) -> bool:
    """Return whether two rectangles intersect.  Only deal with rectangles that have 0,90,-90, 180, etc. rotation
    Algorithm:
    If r1 and r2 both refer to the same rectangle, return False.
    Do each axis separately.  Return true when result from both axes returns true, otherwise return false.
    X axis yields false only when both left and right of one rect are greater or less than both
    left and right of the other rect.  Same with Y axis and top/bottom.
    |--------|
                |--------| -> false
    anything else -> true for each axis
    """
    # rot1 = xfrm_to_deg(r1.transform)
    # rot2 = xfrm_to_deg(r2.transform)
    # assert(rot1 % 90 == 0)
    # assert(rot2 % 90 == 0)

    if r1 is r2: return False

    # r1 clears r2 on the left
    if r1.right <= r2.left: return False

    # r1 clears r2 on the right
    if r1.left >= r2.right: return False

    # r1 clears r2 on the top
    if r1.bottom >= r2.top: return False

    # r1 clears r2 on the bottom
    if r1.top <= r2.bottom: return False

    # Some overlap must exist
    # Return the amount of overlap as [left, right, bottom, top]
    # where the value indicates the amount that the given edge of 
    # r1 overlaps into r2.
    
    retval = []
    # left edge of r1 inside r2
    if r2.left < r1.left < r2.right:
        retval.append(r1.left - r2.left) # a positive number
    else:
        retval.append(None)
    
    # right edge of r1 inside r2
    if r2.left < r1.right < r2.right:
        retval.append(r1.right - r2.left)
    else:
        retval.append(None)

    # bottom edge of r1 inside r2
    if r2.bottom < r1.bottom < r2.top:
        retval.append(r1.bottom - r2.bottom)
    else:
        retval.append(None)

    # top edge of r1 inside r2
    if r2.bottom < r1.top < r2.top:
        retval.append(r1.top - r2.bottom)
    else:
        retval.append(None)

    return retval

def make_packages(qty=1, minpins=2, maxpins=10):
    """Generate random packages.  Return iterable"""
    # ru = random.uniform
    rs = random.sample
    return [Package(name=rs(packspecs.keys(), 1)[0]) for i in range(qty)]

    # return (Package('RCMF01005', pos=xy(0.0,   0.0), rot=random.uniform(-2,2)),
    #         Package('RCMF01005', pos=xy(2.5,   0.0), rot=random.uniform(-2,2)),
    #         Package('RCMF0201' , pos=xy(0.0,   1.5), rot=random.uniform(-2,2)),
    #         Package('RCMF0201' , pos=xy(2.5,   1.5), rot=random.uniform(-2,2)),
    #         Package('RCMF0402' , pos=xy(0.0,   3.0), rot=random.uniform(-2,2)),
    #         Package('RCMF0402' , pos=xy(2.5,   3.0), rot=random.uniform(-2,2)),
    #         Package('RCMF0603' , pos=xy(0.0,   4.5), rot=random.uniform(-2,2)),
    #         Package('RCMF0603' , pos=xy(2.5,   4.5), rot=random.uniform(-2,2)),
    #         Package('RCMF0805' , pos=xy(0.0,   6.0), rot=random.uniform(-2,2)),
    #         Package('RCMF0805' , pos=xy(3.5,   6.0), rot=random.uniform(-2,2)),
    #         Package('RCMF1206' , pos=xy(0.0,   8.0), rot=random.uniform(-2,2)),
    #         Package('RCMF1206' , pos=xy(5.0,   8.0), rot=random.uniform(-2,2)),
    #         Package('RCMF1210' , pos=xy(0.0,  11.0), rot=random.uniform(-2,2)),
    #         Package('RCMF1210' , pos=xy(6.0,  11.0), rot=random.uniform(-2,2)),
    #         Package('RCMF2010' , pos=xy(0.0,  15.0), rot=random.uniform(-2,2)),
    #         Package('RCMF2010' , pos=xy(7.0,  15.0), rot=random.uniform(-2,2)),
    #         Package('RCMF2512' , pos=xy(0.0,  19.0), rot=random.uniform(-2,2)),
    #         Package('RCMF2512' , pos=xy(10.0, 19.0), rot=random.uniform(-2,2)),
    #         Package('LT4312f'  , pos=xy(10.0,  2.0), rot=random.uniform(-2,2)),
    #         Package('SX9376'   , pos=xy(9.0,   7.0), rot=random.uniform(-2,2)),
    #         Package('LPS22DF'  , pos=xy(12.0,  7.0), rot=random.uniform(-2,2)),
    #         Package('SLG51001' , pos=xy(10.0, 10.0), rot=random.uniform(-2,2)))


def randomize_packages(packages: list) -> None:
    for p in packages:
        rx = random.uniform(-1, 1)
        ry = random.uniform(-1, 1)
        p.translate_to(xy(rx, ry))

def draw_packages(ax: plt.Axes, packages: list) -> None:
    for package in packages:
        package.draw(ax)
    plt.pause(0.0001)
    plt.draw()

def packages_intersect(packages: list) -> bool:
    # Returns the set of packages that have at least one
    # overlap with another package
    result = set()
    for p1 in packages:
        for p2 in packages:
            iter_val = bbox_intersect(p1.bbox, p2.bbox)
            if iter_val: 
                result.add(p1)
                result.add(p2)
    return result or False

def separate_packages1(packages: list) -> None:
    # Move packages outward along the vector from the centroid
    rpos = [pkg.pos for pkg in packages]
    centroidx = np.average([p.x for p in rpos])
    centroidy = np.average([p.y for p in rpos])
    centroid = xy(centroidx, centroidy)
    for pkg in packages:
        ctrvec = pkg.pos - centroid
        pkg.translate(ctrvec)

def separate_packages2(packages: list) -> None:
    # Move packages by repulsive field
    rpos = [pkg.pos for pkg in packages]
    k = 0.1
    for pkg1 in packages:
        pkg1force = []
        for pkg2 in packages:
            if pkg1 is pkg2: continue
            ptdiff = pkg1.pos - pkg2.pos
            normvec = ptdiff / ptdiff.mag()
            pushvec = (normvec / ptdiff.magsquared()) * pkg2.bbox.area()
            pkg1force.append(pushvec)
        xforce = k * np.sum([f.x for f in pkg1force])
        yforce = k * np.sum([f.y for f in pkg1force])
        pkg1.translate(xy(xforce, yforce))

def separate_packages3(packages: list, moved: dict = {}) -> None:
    # Move packages by overlap
    for pkg1 in packages:
        for pkg2 in packages:
            if pkg1 is pkg2: continue
            ovl = bbox_intersect(pkg1.bbox, pkg2.bbox)
            if not ovl: continue
            olist = [o for o in ovl if o]
            if not olist: continue
            mv = min(olist)
            if   ovl[0] == mv: amt = xy(-mv, 0.0)
            elif ovl[1] == mv: amt = xy( mv, 0.0)
            elif ovl[2] == mv: amt = xy(0.0, -mv)
            elif ovl[3] == mv: amt = xy(0.0,  mv)
            if pkg2 in moved:
                newiter = moved[pkg2][1] + 1
                newamt = amt + xy(random.uniform(-0.1,0.1), random.uniform(-0.1,0.1)) * newiter
                moved[pkg2] = [newamt, newiter]
            else:
                moved[pkg2] = [amt, 1]
            pkg2.translate(moved[pkg2][0])
    return moved



def compact_packages(packages: list)-> None:
    # Move packages by attractive springs (proportional to distance)
    # Moving by attractive gravity (proportional to 1/distance ^2) wasn't as well behaved
    rpos = [pkg.pos for pkg in packages]
    k = 0.001
    translations = {}
    for pkg1 in packages:
        pkg1force = []
        for pkg2 in packages:
            if pkg1 is pkg2: continue
            ptdiff = pkg1.pos - pkg2.pos
            pushvec = ptdiff * (-k) * pkg2.bbox.area()
            pkg1force.append(pushvec)
        xforce = np.sum([f.x for f in pkg1force])
        yforce = np.sum([f.y for f in pkg1force])
        force = xy(xforce, yforce)
        translations[pkg1] = copy.copy(force)
    
    totalxlate = 0.0
    for pkg, xlate in translations.items():
        pkg.translate(xlate)
        totalxlate = totalxlate + xlate.mag()
    return totalxlate
    

def clr_plot(ax):
    plt.cla()
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    #ax.autoscale(True)
    ax.set_aspect('equal')
    #ax.grid('both')

def on_move(event):
    # get the x and y pixel coords
    x, y = event.x, event.y
    if event.inaxes:
        ax = event.inaxes  # the axes instance
        print('data coords %f %f' % (event.xdata, event.ydata))

bindind_id = None
def on_click(event):
    if event.button is MouseButton.LEFT:
        print('disconnecting callback')
        plt.disconnect(binding_id)

if __name__ == '__main__':

    packages = make_packages(10)

    # packages=(
    #              Package('LPS22DF', pos=xy(-0.5,   0.0), rot=0),
    #              Package('LPS22DF', pos=xy(0.5,   0.0), rot=0),
    #              Package('LT4312f',  pos=xy(-0.0,  0.0), rot=0),
                 # Package('SX9376',  pos=xy(2.5,   0.0), rot=0),
    #            Package('SLG51001',  pos=xy(2.5,   0.0), rot=0),
    #             )
    
    ax = plt.axes()

    binding_id = plt.connect('motion_notify_event', on_move)
    plt.connect('button_press_event', on_click)

    positions = []
    randomize_packages(packages)

    clr_plot(ax)
    draw_packages(ax, packages)
    #plt.pause(1)
    i = 0
    moved_history = {}
    print('starting')
    while (i < 100):
        xsect = packages_intersect(packages) 
        if not xsect: break
    #   print(f'i={i}, intersection qty={len(xsect)}')
        moved_history = separate_packages3(xsect, moved_history)
        clr_plot(ax)
        draw_packages(ax, packages)
        i += 1
        
    print('done',i)

    # xltmag = 1000.0
    # while (xltmag > 0.001):
    #     xltmag = compact_packages(packages)
    #     clr_plot(ax)
    #     draw_packages(ax, packages)

    clr_plot(ax)
    draw_packages(ax, packages)

    plt.show()

