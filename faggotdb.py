from vapoursynth import core, VideoNode, GRAY, YUV
import vapoursynth as vs
import kagefunc as kgf #https://github.com/Irrational-Encoding-Wizardry/kagefunc/
import fvsfunc as fvf #https://github.com/Irrational-Encoding-Wizardry/fvsfunc/
from vsutil import split #https://github.com/Irrational-Encoding-Wizardry/vsutil/blob/master/vsutil.py#L113

core = vs.core

def faggotdb(clip: vs.VideoNode, thrY=40, thrC=None, radiusY=12, radiusC=12, mask="retinex", CbY=44, CrY=44, CbC=44, CrC=44, grainY=15, grainc=0, sample_mode=sample_mode, sample_mode=2, dynamic_grainY=False, dynamic_grainC=False, tv_range=True, binarize=True, binarize_thr=5000) -> vs.VideoNode:

    # Original Idea: Author who created Fag3kdb. Edited by AlucardSama04

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("FaggotDb: This is not a clip")

    if clip.format.bits_per_sample != 16:
        raise TypeError("FaggotDb: Only 16Bit clips are supported")

    if thrC is None:
        thrC = int(round(thrY / 2))

    if grainC is None:
        grainC = int(round(grainY / 2))

    if mask in [-1]:
        mask = clip
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
         raise ValueError("Unknown Mask Mode")

    if binarize:
        mask = core.std.Binarize(mask, threshold=binarize_thr)

    Y, U, V = split(clip)
    U = core.f3kdb.Deband(U, range=radiusC, y=thrC, cb=CbC, cr=CrC, grainy=grainC, grainc=0, sample_mode=sample_mode, dynamic_grain=dynamic_grainC, keep_tv_range=tv_range, output_depth=16)

    V = core.f3kdb.Deband(V, range=radiusC, y=thrC, cb=CbC, cr=CrC, grainy=grainC, grainc=0, sample_mode=sample_mode, dynamic_grain=dynamic_grainC, keep_tv_range=tv_range, output_depth=16)

    filtered = core.std.ShufflePlanes([clip,U,V], [0,0,0], vs.YUV)
    filtered = core.f3kdb.Deband(filtered, range=radiusY, y=thrY, cb=CbY, cr=CrY, grainy=grainY, grainc=0, sample_mode=sample_mode, dynamic_grain=dynamic_grainY, keep_tv_range=tv_range, output_depth=16)

    return core.std.MaskedMerge(filtered, clip, mask)
