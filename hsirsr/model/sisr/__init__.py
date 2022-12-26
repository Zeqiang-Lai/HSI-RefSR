

def sspsr(sf, n_colors):
    from .sspsr.sspsr import SSPSR
    net = SSPSR(n_subs=8, n_ovls=2, n_colors=n_colors, n_blocks=6,
                n_feats=256, n_scale=sf, res_scale=0.1, use_share=True)
    return net


def mcnet(sf, n_colors, stat_path):
    from .mcnet import MCNet
    net = MCNet(n_colors=n_colors, n_feats=64, n_conv=1, upscale_factor=sf, stat_path=stat_path)
    return net


def qrnn3d():
    from .qrnn3d.utils import QRNNREDC3D
    net = QRNNREDC3D(1, 16, 5, [1, 3])
    return net


def biqrnn3d():
    from .biqrnn3d.biqrnn3d_fast import BiFQRNNREDC3D
    net = BiFQRNNREDC3D(1, 16, 5, [1, 3])
    return net
