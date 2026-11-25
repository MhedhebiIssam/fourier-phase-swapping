import torch


def fft(t):
    F = torch.fft.fft2(t, dim=(-2, -1))
    F = torch.fft.fftshift(F, dim=(-2, -1))
    magnitude = torch.abs(F)
    phases = torch.angle(F)
    return magnitude, phases


def ifft(magnitude, phase):
    F = torch.polar(magnitude, phase)
    F = torch.fft.ifftshift(F, dim=(-2, -1))
    reconstructed = torch.fft.ifft2(F, dim=(-2, -1))
    return reconstructed.real
