"""
pytorch-based forward diffraction propagation

From: https://github.com/kqwang/DLPR/
@author: Kaiqiang Wang
Email: kqwang.optics@gmail.com
"""
import numpy as np
import torch
import math
from torch.fft import fft2, ifft2, fftshift, ifftshift

' Forward diffraction propagation in pytorch (cuda)'
def propagation(P, A = np.ones((256, 256)), prop_dis=20, norm=False, dim = 256):
    A = torch.from_numpy(A).cuda().unsqueeze(0).unsqueeze(0)  # amplitude from numpy to torch(cuda)
    U_O = A * torch.exp(1j * P)  # complex-valued amplitudes on object plane
    lamb = 632.8e-6  # wavelength
    pixel_size = 8e-3  # pixel size
    pixel_num = dim  # pixel number
    len = pixel_num * pixel_size  # field of view length
    k = 2 * math.pi / lamb  # wave number

    # frequency domain sampling
    f = 1 / len
    f = f * np.linspace(-dim / 2, dim / 2 - 1, dim)
    [Fx, Fy] = np.meshgrid(f, f)

    # propagation function
    H = torch.exp(1j * k * prop_dis * torch.sqrt(torch.tensor(1 - lamb * lamb * (Fx * Fx + Fy * Fy)))).cuda()

    # propagation from Object plane to Imaging plane
    UO_FFT = ifftshift(fft2(fftshift(U_O)))
    UI = ifftshift(ifft2(fftshift(UO_FFT * H)))

    # get hologram on the Imaging plane
    H = torch.abs(UI) * torch.abs(UI)

    # normalize the hologram to [0,1]
    if norm:
        H_max = H.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)
        H_min = H.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).repeat(1, 1, dim, dim)
        H = torch.div((H - H_min), (H_max - H_min))

    return H