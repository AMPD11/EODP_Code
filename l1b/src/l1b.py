
# LEVEL-1B MODULE

from l1b.src.initL1b import initL1b
from common.io.writeToa import writeToa, readToa
from common.src.auxFunc import getIndexBand
from common.io.readFactor import readFactor, EQ_MULT, EQ_ADD, NC_EXT
import numpy as np
import os
import matplotlib.pyplot as plt

class l1b(initL1b):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1B Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the ISM in Digital Numbers
            # -------------------------------------------------------------------------------
            toa = readToa(self.indir, self.globalConfig.ism_toa + band + '.nc')

            # Equalization (radiometric correction)
            # -------------------------------------------------------------------------------
            if self.l1bConfig.do_equalization:
                self.logger.info("EODP-ALG-L1B-1010: Radiometric Correction (equalization)")

                # Read the multiplicative and additive factors from auxiliary/equalization/
                eq_mult = readFactor(os.path.join(self.auxdir,self.l1bConfig.eq_mult+band+NC_EXT),EQ_MULT)
                eq_add = readFactor(os.path.join(self.auxdir,self.l1bConfig.eq_add+band+NC_EXT),EQ_ADD)

                # Do the equalization and save to file
                toa = self.equalization(toa, eq_add, eq_mult)
                writeToa(self.outdir, self.globalConfig.l1b_toa_eq + band, toa)
                self.plotL1bEq(toa, self.outdir, band)

            # Restitution (absolute radiometric gain)
            # -------------------------------------------------------------------------------
            self.logger.info("EODP-ALG-L1B-1020: Absolute radiometric gain application (restoration)")
            toa = self.restoration(toa, self.l1bConfig.gain[getIndexBand(band)])

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeToa(self.outdir, self.globalConfig.l1b_toa + band, toa)
            self.plotL1bToa(toa, self.outdir, band)

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1B Module!")


    def equalization(self, toa, eq_add, eq_mult):
        """
        Equlization. Apply an offset and a gain.
        :param toa: TOA in DN
        :param eq_add: Offset in DN
        :param eq_mult: Gain factor, adimensional
        :return: TOA in DN, equalized
        """
        # if eq_add.ndim == 1:  eq_add = eq_add[np.newaxis, :]
        # if eq_mult.ndim == 1: eq_mult = eq_mult[np.newaxis, :]
        # return (toa - eq_add) / eq_mult
        return toa

    def restoration(self,toa,gain):
        """
        Absolute Radiometric Gain - restore back to radiances
        :param toa: TOA in DN
        :param gain: gain in [rad/DN]
        :return: TOA in radiances [mW/sr/m2]
        """
        # toa = toa * float(gain)
        self.logger.debug('Sanity check. TOA in radiances after gain application ' + str(toa[1,-1]) + ' [mW/m2/sr]')
        return toa

    def _save_heatmap(self, img, path, title, cbar_label):
        plt.figure()
        plt.imshow(img, origin="lower")
        plt.colorbar(label=cbar_label)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def _save_col_profile(self, img, path, title, y_label):
        prof = img.mean(axis=0)  # media por columnas
        x = np.arange(prof.size)
        plt.figure()
        plt.plot(x, prof)
        plt.title(title)
        plt.xlabel("act_columns")
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    def plotL1bToa(self, toa_l1b, outputdir, band):
        """PNG del producto restaurado (radiancias)."""
        idx = band.split('-')[-1]  # "0","1","2","3"
        heatmap = os.path.join(outputdir, f"l1b_toa_VNIR-{idx}.png")
        profpng = os.path.join(outputdir, f"l1b_toa_VNIR-{idx}_profile.png")
        self._save_heatmap(toa_l1b, heatmap, f"L1B TOA {band}", "TOA [mW/m²/sr]")
        self._save_col_profile(toa_l1b, profpng, f"L1B TOA column-mean {band}", "TOA [mW/m²/sr]")

    def plotL1bEq(self, toa_eq, outputdir, band):
        """PNG del intermedio equalizado (DN)."""
        idx = band.split('-')[-1]
        heatmap = os.path.join(outputdir, f"l1b_toa_eq_VNIR-{idx}.png")
        profpng = os.path.join(outputdir, f"l1b_toa_eq_VNIR-{idx}_profile.png")
        self._save_heatmap(toa_eq, heatmap, f"Equalized TOA {band}", "DN")
        self._save_col_profile(toa_eq, profpng, f"Equalized TOA column-mean {band}", "DN")
