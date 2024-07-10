"""
pytorch-based forward diffraction propagation

From: https://github.com/kqwang/DLPR/
@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import numpy as np
import torch
import torch.nn.functional as F
import math
from torch.fft import fft2, ifft2, fftshift, ifftshift

from matplotlib import pyplot as plt

' Forward diffraction propagation in pytorch (cuda)'
def propagation(P,  prop_dis=20, norm=False, dim = 256, pad=None):
    A = np.ones(dim, dim)
    A = torch.from_numpy(A).cuda().unsqueeze(0).unsqueeze(0)  # amplitude from numpy to torch(cuda)

    # avoid edge diffraction effects through padding
    if pad:
        p2d = (int(dim/4), int(dim/4), int(dim/4), int(dim/4))  # pad last dim and 2nd to last
        A = F.pad(A, p2d, "replicate") # pad A
        P = F.pad(P, p2d, "replicate") # pad P

        # P_np = P[0, 0, :, :].cpu().numpy()
        # plt.imshow(P_np, cmap="inferno")
        # plt.title('phase (gt)')
        # plt.colorbar(fraction=0.05, pad=0.05)
        # plt.axis('off')
        # plt.show()


    U_O = A * torch.exp(1j * P)  # complex-valued amplitudes on object plane
    lamb = 632.8e-6  # wavelength
    pixel_size = 8e-3  # pixel size
    pixel_num = dim  # pixel number
    if pad:
        pixel_num = dim + int(dim / 2)
    len = pixel_num * pixel_size  # field of view length
    k = 2 * math.pi / lamb  # wave number

    # frequency domain sampling
    f = 1 / len
    f = f * np.linspace(-pixel_num / 2, pixel_num / 2 - 1, pixel_num)
    [Fx, Fy] = np.meshgrid(f, f)

    # propagation function
    H = torch.exp(1j * k * prop_dis * torch.sqrt(torch.tensor(1 - lamb * lamb * (Fx * Fx + Fy * Fy)))).cuda()

    # propagation from Object plane to Imaging plane
    UO_FFT = ifftshift(fft2(fftshift(U_O)))
    UI = ifftshift(ifft2(fftshift(UO_FFT * H)))

    # get hologram on the Imaging plane
    H = torch.abs(UI) * torch.abs(UI)

    # take the middle area, if padding is used
    if pad:
        # H_np = H[0, 0, :, :].cpu().numpy()
        # plt.imshow(H_np, cmap="gray")
        # plt.title('Holo')
        # plt.colorbar(fraction=0.05, pad=0.05)
        # plt.axis('off')
        # plt.show()
        H = H[:, :, int(dim/4):(dim+int(dim/4)), int(dim/4):(dim+int(dim/4))]

    # normalize the hologram to [0,1]
    if norm:
        H_max = H.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)
        H_min = H.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)
        H = torch.div((H - H_min), (H_max - H_min))

    return H