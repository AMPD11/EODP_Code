from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = 1 # dummy

        # Plot cuts ACT/ALT of the MTF
        self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        # 1D frequency axes in cycles/pixel
        frAct = np.fft.fftshift(np.fft.fftfreq(ncolumns, d=1.0))
        frAlt = np.fft.fftshift(np.fft.fftfreq(nlines, d=1.0))

        # Normalize to Nyquist (0.5 cycles/pixel)
        fnyq = 0.5
        fnAct = frAct / fnyq
        fnAlt = frAlt / fnyq

        # 2D grids (relative and normalized)
        FrX, FrY = np.meshgrid(frAct, frAlt)
        FnX, FnY = np.meshgrid(fnAct, fnAlt)

        # Radial frequency
        fr2D = np.sqrt(FrX ** 2 + FrY ** 2)  # cycles/pixel
        fn2D = np.sqrt(FnX ** 2 + FnY ** 2)  # normalized to Nyquist
        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        nu = np.clip(fr2D, 0.0, None).astype(np.float64)
        Hdiff = np.zeros_like(nu, dtype=np.float64)

        mask = nu <= 1.0
        x = nu[mask]
        Hdiff[mask] = (2.0 / np.pi) * (np.arccos(x) - x * np.sqrt(1.0 - x * x))
        return Hdiff

    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        nu = np.asarray(fr2D, dtype=np.float64)
        # Scale factor: defocus is given as defocus/(f/N). For low values (0–2),
        # a quadratic attenuation in frequency is adequate.
        alpha = float(defocus)
        Hdefoc = np.exp(-(np.pi * alpha * nu) ** 2)
        # Numerical safety and bounds
        Hdefoc = np.clip(Hdefoc, 0.0, 1.0)
        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        nu = np.asarray(fr2D, dtype=np.float64)

        # Phase RMS (dimensionless) for each regime
        aLF = 2.0 * np.pi * (float(wLF) / float(lambd))
        aHF = 2.0 * np.pi * (float(wHF) / float(lambd))

        # Empirical attenuation; kLF/kHF scale how fast the MTF decays with frequency
        Hlf = np.exp(-(kLF * aLF * nu) ** 2)
        Hhf = np.exp(-(kHF * aHF * nu) ** 2)

        Hwfe = Hlf * Hhf
        Hwfe = np.clip(Hwfe, 0.0, 1.0)
        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        fn = np.asarray(fn2D, dtype=np.float64)
        Hdet = np.sinc(fn)  # = sin(pi*fn)/(pi*fn), with Hdet(0)=1
        Hdet = np.clip(Hdet, 0.0, 1.0)
        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """
        fnAlt = np.asarray(fnAlt, dtype=np.float64)
        frAlt = 0.5 * fnAlt  # cycles/pixel
        Halt = np.sinc(ksmear * frAlt)  # sin(pi*x)/(pi*x)
        Halt = np.clip(Halt, 0.0, 1.0)

        Hsmear = np.tile(Halt[:, None], (1, ncolumns))
        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        fn = np.asarray(fn2D, dtype=np.float64)
        fr = 0.5 * fn  # cycles/pixel
        Hmotion = np.sinc(kmotion * fr)
        Hmotion = np.clip(Hmotion, 0.0, 1.0)
        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """
        #TODO


