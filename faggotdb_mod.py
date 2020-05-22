#from vapoursynth import core, VideoNode, GRAY, YUV
import vapoursynth as vs
import fvsfunc as fvf # https://github.com/Irrational-Encoding-Wizardry/fvsfunc/blob/master/fvsfunc.py
from vsutil import plane # https://github.com/Irrational-Encoding-Wizardry/vsutil/blob/master/vsutil.py
import kagefunc as kgf #https://github.com/Irrational-Encoding-Wizardry/kagefunc/

core = vs.core

def FaggotDB(clip: vs.VideoNode, thrY=40, thrC=None, radiusY=15, radiusC=15, CbY=44, CrY=44, CbC=44, CrC=44, grainY=20, grainC=None, sample_mode=2, neo=False, dynamic_grainY=False, dynamic_grainC=False, tv_range=True, mask=None) -> vs.VideoNode:

    funcName = "FaggotDB"

    # Original Idea: Author who created Fag3kdb. Edited by AlucardSama04

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f"{funcName}: This is not a clip")

    if clip.format.bits_per_sample != 16:
        raise TypeError(f"{funcName}: Only 16Bit clips are supported")

    if not isinstance(mask, vs.VideoNode):
        raise vs.Error(f"{funcName}: mask' only clip inputs")

    if thrC is None:
        thrC = int(round(thrY / 2))

    if grainC is None:
        grainC = int(round(grainY / 2))

    f3kdb = core.neo_f3kdb.Deband if neo else core.f3kdb.Deband

    U = plane(clip, 1)
    V = plane(clip, 2)
    U = f3kdb(U, range=radiusC, y=thrC, cb=CbC, cr=CrC, grainy=grainC, grainc=0, sample_mode=sample_mode, dynamic_grain=dynamic_grainC, keep_tv_range=tv_range, output_depth=16)

    V = f3kdb(V, range=radiusC, y=thrC, cb=CbC, cr=CrC, grainy=grainC, grainc=0, sample_mode=sample_mode, dynamic_grain=dynamic_grainC, keep_tv_range=tv_range, output_depth=16)

    filtered = core.std.ShufflePlanes([clip,U,V], [0,0,0], vs.YUV)
    filtered = f3kdb(filtered, range=radiusY, y=thrY, cb=CbY, cr=CrY, grainy=grainY, grainc=0, sample_mode=sample_mode, dynamic_grain=dynamic_grainY, keep_tv_range=tv_range, output_depth=16)

    return core.std.MaskedMerge(filtered, clip, mask)#from vapoursynth import core, VideoNode, GRAY, YUV

def faggotdb_mod(clip: vs.VideoNode, thrY=40, thrC=None, radiusY=15, radiusC=15, CbY=44, CrY=44, CbC=44, CrC=44, grainY=32, grainC=None, grainCC=0, sample_mode=2, neo=True, dynamic_grainY=False, dynamic_grainC=False, tv_range=True, mask="retinex", binarize=False, binarize_thr=70, grayscale=True, bitresamp=False, outbits=None, blurmask=True, horizblur=2, vertblur=2) -> vs.VideoNode:

    funcName = "faggotdb_mod" # name kept to be fallback compatible

    # Original Idea: Author who created Fag3kdb. Edited by AlucardSama04; additional modifications by l00t

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f"{funcName}: This is not a clip")
        
    if outbits is None:
        outbits = clip.format.bits_per_sample

    if bitresamp:
        if clip.format.bits_per_sample != outbits:
            clip = fvf.Depth(clip, bits=outbits) # instead of error, auto convert to 16 bits
    elif clip.format.bits_per_sample != outbits:
        raise TypeError(f"{funcName}: Input-output bitdepth mismatch")

    # if not isinstance(mask, vs.VideoNode):
        # raise vs.Error(f"{funcName}: mask' only clip inputs")
        
    if mask in [-1]: # more user friendly if we do the masking intentionally
        mask = clip
        blurmask = False
    elif mask in [0, "retinex"]:
        mask = kgf.retinex_edgemask(clip)
    elif mask in [1, "kirsch"]:
        mask = kgf.kirsch(clip)
    elif mask in [2, "Sobel"]:
        mask = core.std.Sobel(clip, scale=1)
    elif mask in [3, "Prewitt"]:
        mask = core.std.Prewitt(clip)
    elif mask in [4, "GF"]:
        mask = fvf.GradFun3(clip, mask=2, debug=1)
    else:
         raise ValueError(f"{funcName}: Unknown Mask Mode")

    if grayscale:
        mask = core.std.ShufflePlanes(mask, planes=0, colorfamily=vs.GRAY)
         
    if binarize:     # binarize treshold should be adjusted according to bitdepth
        mask = core.std.Binarize(mask, threshold=binarize_thr)
        
    if blurmask:
        mask = core.std.BoxBlur(mask, hradius=horizblur, vradius=vertblur)

    if thrC is None:
        thrC = int(round(thrY / 2))

    if grainC is None:
        grainC = int(round(grainY / 2))

    if grainCC is None:
        grainCC = 0
        
    f3kdb = core.neo_f3kdb.Deband if neo else core.f3kdb.Deband

    U = plane(clip, 1)
    V = plane(clip, 2)
    
    U = f3kdb(U, range=radiusC, y=thrC, cb=CbC, cr=CrC, grainy=grainC, grainc=0, sample_mode=sample_mode, dynamic_grain=dynamic_grainC, keep_tv_range=tv_range, output_depth=outbits)
    
    V = f3kdb(V, range=radiusC, y=thrC, cb=CbC, cr=CrC, grainy=grainC, grainc=0, sample_mode=sample_mode, dynamic_grain=dynamic_grainC, keep_tv_range=tv_range, output_depth=outbits)

    filtered = core.std.ShufflePlanes([clip,U,V], [0,0,0], vs.YUV)
    
    filtered = f3kdb(filtered, range=radiusY, y=thrY, cb=CbY, cr=CrY, grainy=grainY, grainc=grainCC, sample_mode=sample_mode, dynamic_grain=dynamic_grainY, keep_tv_range=tv_range, output_depth=outbits) # if grainCC > 0 UV planes will be debanded once again

    return core.std.MaskedMerge(filtered, clip, mask)
