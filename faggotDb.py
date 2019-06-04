from vapoursynth import core, VideoNode, GRAY, YUV
import vapoursynth as vs
import kagefunc as kgf #https://github.com/Irrational-Encoding-Wizardry/kagefunc/
import fvsfunc as fvf #https://github.com/Irrational-Encoding-Wizardry/fvsfunc/

core = vs.core

def FaggotDb(clip: vs.VideoNode, thrY=40, thrC=None, radiusY=12, radiusC=12, mask="retinex", CbY=44, CrY=44, CbC=44, CrC=44, grainY=15, grainC=0, dynamic_grainY=False, dynamic_grainC=False, tv_range=True) -> vs.VideoNode:

    # Original Idea: Author that created Fag3kdb. Edited by AlucardSama04 (Added Custom Masks)

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("FaggotDb: This is not a clip")

    if clip.format.bits_per_sample != 16:
        raise TypeError("FaggotDb: Only 16Bit clips are supported")

    if thrC is None:
        thrC = thrY / 2

    if grainC is None:
        grainC = int(round(grainY / 2))

    if mask in [0, "retinex"]:
        mask = kgf.retinex_edgemask(clip).std.Binarize(5000)
    elif mask in [1, "kirsch"]:
        mask = kgf.kirsch(clip).std.Binarize(5000)
    elif mask in [2, "FSobel"]:
        mask = kgf.fast_sobel(clip).std.Binarize(5000)
    elif mask in [3, "Sobel"]:
        mask = core.std.Sobel(clip, scale=1).std.Binarize(5000)
    elif mask in [4, "Prewitt"]:
        mask = core.std.Prewitt(clip).std.Binarize(5000)
    elif mask in [5, "GF"]:
        mask = fvf.GradFun3(clip, mask=2, debug=1).std.Binarize(5000)
    else:
         raise ValueError("Unknown Mask Mode")

    U = core.std.ShufflePlanes(clip, 1, GRAY)
    U = core.f3kdb.Deband(U, range=radiusC, y=thrC, cb=CbC, cr=CrC, grainy=grainC, grainc=0, dynamic_grain=dynamic_grainC, keep_tv_range=tv_range, output_depth=16)

    V = core.std.ShufflePlanes(clip, 2, GRAY)
    V = core.f3kdb.Deband(V, range=radiusC, y=thrC, cb=CbC, cr=CrC, grainy=grainC, grainc=0, dynamic_grain=dynamic_grainC, keep_tv_range=tv_range, output_depth=16)

    filtered = core.std.ShufflePlanes([clip,U,V], [0,0,0], vs.YUV)
    filtered = core.f3kdb.Deband(filtered, range=radiusY, y=thrY, cb=CbY, cr=CrY, grainy=grainY, grainc=0, dynamic_grain=dynamic_grainY, keep_tv_range=tv_range, output_depth=16)

    return core.std.MaskedMerge(filtered, clip, mask)
