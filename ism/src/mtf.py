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
        # Ensure all are float64 and within [0,1]
        Hdiff = np.asarray(Hdiff, dtype=np.float64)
        Hdefoc = np.asarray(Hdefoc, dtype=np.float64)
        Hwfe = np.asarray(Hwfe, dtype=np.float64)
        Hdet = np.asarray(Hdet, dtype=np.float64)
        Hsmear = np.asarray(Hsmear, dtype=np.float64)
        Hmotion = np.asarray(Hmotion, dtype=np.float64)

        # Shapes must all match (nlines, ncolumns)
        if not (Hdiff.shape == Hdefoc.shape == Hwfe.shape == Hdet.shape == Hsmear.shape == Hmotion.shape):
            raise ValueError(
                f"MTF contributors have mismatched shapes: "
                f"diff{Hdiff.shape}, def{Hdefoc.shape}, wfe{Hwfe.shape}, det{Hdet.shape}, "
                f"smear{Hsmear.shape}, mot{Hmotion.shape}"
            )

        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion
        Hsys = np.clip(Hsys, 0.0, 1.0)

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
        # 1) Discrete frequency axes in cycles/pixel (un-normalized to Nyquist)
        fr_act_pix = np.fft.fftshift(np.fft.fftfreq(ncolumns, d=1.0))  # cycles/pixel
        fr_alt_pix = np.fft.fftshift(np.fft.fftfreq(nlines, d=1.0))  # cycles/pixel

        # 2) Normalized-to-Nyquist (Nyquist = 0.5 cycles/pixel)
        fnyq = 0.5
        fnAct = fr_act_pix / fnyq
        fnAlt = fr_alt_pix / fnyq

        # 3) Build 2-D grids
        FRx, FRy = np.meshgrid(fr_act_pix, fr_alt_pix)  # cycles/pixel
        FNx, FNy = np.meshgrid(fnAct, fnAlt)  # f / f_Nyq

        # 4) Magnitudes
        f_pix = np.sqrt(FRx ** 2 + FRy ** 2)  # cycles/pixel
        fn2D = np.sqrt(FNx ** 2 + FNy ** 2)  # normalized to Nyquist

        # 5) Convert to relative-to-cutoff: fr2D = (f / f_c)
        #    spatial freq [cycles/m] = (cycles/pixel) / w
        f_spatial = f_pix / float(w)  # cycles/m
        f_c = float(D) / (float(lambd) * float(focal))  # cycles/m
        fr2D = f_spatial / f_c  # dimensionless

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        nu = np.asarray(fr2D, dtype=np.float64)
        nu = np.clip(nu, 0.0, None)
        Hdiff = np.zeros_like(nu, dtype=np.float64)
        m = nu <= 1.0
        x = nu[m]
        Hdiff[m] = (2.0 / np.pi) * (np.arccos(x) - x * np.sqrt(1.0 - x * x))
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

        # Phase RMS in waves (no 2π): φ_rms = w/λ
        aLF = float(wLF) / float(lambd)
        aHF = float(wHF) / float(lambd)

        # Empirical attenuation (softer than using 2π)
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
        fn = 0.5 * fn  # cycles/pixel
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

        # Save 2D fields to .nc (same naming as reference)
        # writeMat(dir, "basename_without_ext", array)
        writeMat(directory, f"Hdiff_{band}", Hdiff)
        writeMat(directory, f"Hdefoc_{band}", Hdefoc)
        writeMat(directory, f"Hwfe_{band}", Hwfe)
        writeMat(directory, f"Hdet_{band}", Hdet)
        writeMat(directory, f"Hsmear_{band}", Hsmear)
        writeMat(directory, f"Hmotion_{band}", Hmotion)
        writeMat(directory, f"Hsys_{band}", Hsys)

        # Frequency axes in cycles/pixel (normalized-to-Nyquist axes are fnAct/fnAlt)
        # We want 0..Nyquist (0.5 cycles/pixel)
        fr_act = 0.5 * np.asarray(fnAct, dtype=np.float64)  # shape (ncolumns,)
        fr_alt = 0.5 * np.asarray(fnAlt, dtype=np.float64)  # shape (nlines,)

        # Central indices (DC across the orthogonal axis)
        i_c = nlines // 2
        j_c = ncolumns // 2

        # ACT slice (row center, vary ACT frequency)
        s_diff_act = Hdiff[i_c, :]
        s_defoc_act = Hdefoc[i_c, :]
        s_hwfe_act = Hwfe[i_c, :]
        s_det_act = Hdet[i_c, :]
        s_smear_act = Hsmear[i_c, :]
        s_motion_act = Hmotion[i_c, :]
        s_sys_act = Hsys[i_c, :]

        # ALT slice (column center, vary ALT frequency)
        s_diff_alt = Hdiff[:, j_c]
        s_defoc_alt = Hdefoc[:, j_c]
        s_hwfe_alt = Hwfe[:, j_c]
        s_det_alt = Hdet[:, j_c]
        s_smear_alt = Hsmear[:, j_c]
        s_motion_alt = Hmotion[:, j_c]
        s_sys_alt = Hsys[:, j_c]

        # Keep non-negative half (from DC to Nyquist)
        # fftshift-like layout: DC is in the middle
        def pos_half(axis, y):
            mid = len(axis) // 2
            return axis[mid:], y[mid:]

        x_act, y_diff_act = pos_half(fr_act, s_diff_act)
        _, y_defoc_act = pos_half(fr_act, s_defoc_act)
        _, y_hwfe_act = pos_half(fr_act, s_hwfe_act)
        _, y_det_act = pos_half(fr_act, s_det_act)
        _, y_smear_act = pos_half(fr_act, s_smear_act)
        _, y_motion_act = pos_half(fr_act, s_motion_act)
        _, y_sys_act = pos_half(fr_act, s_sys_act)

        x_alt, y_diff_alt = pos_half(fr_alt, s_diff_alt)
        _, y_defoc_alt = pos_half(fr_alt, s_defoc_alt)
        _, y_hwfe_alt = pos_half(fr_alt, s_hwfe_alt)
        _, y_det_alt = pos_half(fr_alt, s_det_alt)
        _, y_smear_alt = pos_half(fr_alt, s_smear_alt)
        _, y_motion_alt = pos_half(fr_alt, s_motion_alt)
        _, y_sys_alt = pos_half(fr_alt, s_sys_alt)

        # Plot helper
        def plot_one(x, curves, labels, title, fname):
            plt.figure(figsize=(10, 5))
            for y, lab in zip(curves, labels):
                plt.plot(x, y, linewidth=2, label=lab)
            plt.axvline(0.5, linestyle="--", color="k", linewidth=2)  # Nyquist
            plt.xlim(0.0, 0.5)
            plt.ylim(0.0, 1.05)
            plt.xlabel("Spatial frequencies f/(1/w) [-]")
            plt.ylabel("MTF")
            plt.title(title)
            plt.grid(True, alpha=0.25)
            plt.legend(loc="lower left", ncol=2, fontsize=9)
            os.makedirs(directory, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(directory, fname), dpi=200)
            plt.close()

        labels = [
            "Diffraction MTF", "Defocus MTF", "WFE Aberrations MTF",
            "Detector MTF", "Smearing MTF", "Motion blur MTF", "System MTF"
        ]

        plot_one(
            x_act,
            [y_diff_act, y_defoc_act, y_hwfe_act, y_det_act, y_smear_act, y_motion_act, y_sys_act],
            labels,
            f"System MTF slice ACT for {band}",
            f"MTF_ACT_{band}.png"
        )
        plot_one(
            x_alt,
            [y_diff_alt, y_defoc_alt, y_hwfe_alt, y_det_alt, y_smear_alt, y_motion_alt, y_sys_alt],
            labels,
            f"System MTF slice ALT for {band}",
            f"MTF_ALT_{band}.png"
        )

        # MTF at Nyquist (last point)
        mtf_nyq_act = float(y_sys_act[-1]) if len(y_sys_act) else np.nan
        mtf_nyq_alt = float(y_sys_alt[-1]) if len(y_sys_alt) else np.nan

        # Log and save a tiny text report per band
        if self.logger:
            self.logger.info(f"{band}: MTF_Nyquist ACT={mtf_nyq_act:.4f}, ALT={mtf_nyq_alt:.4f}")
        with open(os.path.join(directory, f"MTF_Nyquist_{band}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{band}: MTF_Nyquist ACT={mtf_nyq_act:.6f}, ALT={mtf_nyq_alt:.6f}\n")


