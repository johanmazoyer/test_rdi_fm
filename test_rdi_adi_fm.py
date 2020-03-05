import os
import numpy as np
import astropy.io.fits as fits

from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter

from datetime import datetime

import pyklip.rdi as rdi
import glob
import pyklip.instruments.GPI as GPI
import pyklip.klip as klip
import pyklip.parallelized as parallelized

import astropy.io.fits as fits
from astropy.convolution import convolve
import pyklip.fm as fm

from pyklip.fmlib.diskfm import DiskFM

from fit_2Dgaussian import fitgaussian

def make_phony_disk(dim):
    """
    Create a very simple disk model

    Args:
        dim: Dimension of the array

    Returns:
        centered ellisp disk

    """

    phony_disk = np.zeros((dim, dim))
    PA_rad = np.radians(90+27)  # 27 deg
    incl_rad = np.radians(76)

    # r1 = 10
    # r2 = 15

    r1 = 75
    r2 = 82


    x = np.arange(dim, dtype=np.float)[None, :] - dim // 2
    y = np.arange(dim, dtype=np.float)[:, None] - dim // 2

    x1 = x * np.cos(PA_rad) + y * np.sin(PA_rad)
    y1 = -x * np.sin(PA_rad) + y * np.cos(PA_rad)

    x2 = x1
    y2 = y1 / np.cos(incl_rad)
    rho2dellip = np.sqrt(x2**2 + y2**2)

    phony_disk[np.where((rho2dellip > r1) & (rho2dellip < r2))] = 1
    
    return phony_disk/100.



dir_test = '/Users/jmazoyer/Dropbox/Work/python/python_data/test_smallGPIlib_RDI/larger_test/'

aligned_center = [140.,140.]


lib_files = sorted(glob.glob(dir_test+'*.fits'))
#print data_files
datasetlib = GPI.GPIData(lib_files, highpass=False, quiet = True)
datasetlib.spectral_collapse(collapse_channels=1,align_frames=True, aligned_center=aligned_center)


# make the PSF library
# we need to compute the correlation matrix of all images vs each other since we haven't computed it before
# psflib = rdi.PSFLibrary(datasetlib.input,aligned_center , 
#                             datasetlib.filenames, compute_correlation=True)

# # save the correlation matrix to disk so that we also don't need to recomptue this ever again
# # In the future we can just pass in the correlation matrix into the PSFLibrary object rather 
# # than having it compute it
# psflib.save_correlation(dir_test+"test_results/corr_matrix-SMALLTEST.fits", overwrite=True)

# read in the correlation matrix we already saved
corr_matrix = fits.getdata(dir_test+"test_results/corr_matrix-SMALLTEST.fits")

# make the PSF library again, this time we have the correlation matrix
psflib = rdi.PSFLibrary(datasetlib.input, aligned_center, 
                        datasetlib.filenames, correlation_matrix=corr_matrix)


data_files = sorted(glob.glob(dir_test + 'S20160318S*.fits'))
#print data_files
dataset = GPI.GPIData(data_files, highpass=False, quiet = True)

dataset.generate_psfs(boxrad=8)
instrument_psf = np.mean(dataset.psfs,axis = 0)
instrument_psf[np.where(instrument_psf < np.max(instrument_psf)/100.)] = 0

dataset.spectral_collapse(collapse_channels=1,align_frames=True, aligned_center=aligned_center)



psflib.prepare_library(dataset)

# now we can run RDI klip
# as all RDI images are aligned to aligned_center, we need to pass in that aligned_center into KLIP
numbasis=[2, 4, 8, 10, 20, 30, 37] # number of KL basis vectors to use to model the PSF. We will try several different ones

maxnumbasis=100 # maximum number of most correlated PSFs to do PCA reconstruction with
annuli=1
subsections=1
OWA = 90

dataset.OWA = OWA

# fileprefix="pyklip_parallelizedRDItest"
# parallelized.klip_dataset(dataset, outputdir=dir_test + "test_results/", fileprefix=fileprefix, annuli=annuli,
#                         subsections=subsections, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="RDI",
#                         aligned_center=aligned_center, highpass=False, psf_library=psflib)

# fileprefix="pyklip_parallelizedADItest"
# parallelized.klip_dataset(dataset, outputdir=dir_test + "test_results/", fileprefix=fileprefix, annuli=annuli,
#                         subsections=subsections, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="ADI",
#                         aligned_center=aligned_center, highpass=False)


# create a phony disk model and convovle it by the instrument psf
phony_disk_model = make_phony_disk(281)


model_convolved = convolve(phony_disk_model,
                            instrument_psf,
                            boundary="wrap")


fits.writeto(
    dir_test+ "test_results/psf.fits",
    instrument_psf,
    overwrite=True,
)

fits.writeto(
    dir_test+ "test_results/disk_model_before.fits",
    phony_disk_model,
    overwrite=True,
)

fits.writeto(
    dir_test+ "test_results/disk_model_after.fits",
    model_convolved,
    overwrite=True,
)


print('')
print('')
print('BEGIN FM')
print('')
print('')
print('')

fileprefix="pyklip_FM_ADItest"

diskobj = DiskFM(
            dataset.input.shape,
            numbasis,
            dataset,
            model_convolved,
            basis_filename=os.path.join(dir_test,
                                        "test_results/" + fileprefix + "_KLbasis.h5"),
            save_basis=True,
            aligned_center=aligned_center,
        )

fm.klip_dataset(
    dataset,
    diskobj,
    outputdir=dir_test + "test_results/", fileprefix=fileprefix, annuli=annuli,
                         subsections=subsections, numbasis=numbasis, 
                         maxnumbasis=maxnumbasis, mode="ADI",
                         aligned_center=aligned_center, highpass=False, 
                         mute_progression = True,
)


diskobj = DiskFM(
    dataset.input.shape,
    numbasis,
    dataset,
    model_convolved,
    basis_filename=os.path.join(dir_test,
                                        "test_results/" + fileprefix + "_KLbasis.h5"),
    load_from_basis=True
)



diskobj.update_disk(model_convolved)

then = datetime.now()
modelfm_here = diskobj.fm_parallelized()
print('DiskFM ADI', datetime.now() - then)

return_by_fm_parallelized = modelfm_here[0]  # first KL

fits.writeto(
    dir_test+ "test_results/" + fileprefix + 'returnedbydiskFMparallelized.fits',
    return_by_fm_parallelized,
    overwrite=True,
)


print('')
print('')
print('fin ADI begin RDI')
print('')
print('')
print('')



fileprefix="pyklip_FM_RDItest"

diskobj = DiskFM(
            dataset.input.shape,
            numbasis,
            dataset,
            model_convolved,
            basis_filename=os.path.join(dir_test,
                                        "test_results/" + fileprefix + "_KLbasis.h5"),
            save_basis=True,
            aligned_center=aligned_center
        )

fm.klip_dataset(
    dataset,
    diskobj,
    outputdir=dir_test + "test_results/", fileprefix=fileprefix, annuli=annuli,
                         subsections=subsections, numbasis=numbasis, 
                         maxnumbasis=maxnumbasis, mode="RDI",
                         aligned_center=aligned_center, highpass=False, 
                         mute_progression = True, psf_library=psflib
)

diskobj = DiskFM(
    dataset.input.shape,
    numbasis,
    dataset,
    model_convolved,
    basis_filename=os.path.join(dir_test,
                                        "test_results/" + fileprefix + "_KLbasis.h5"),
    load_from_basis=True
)

diskobj.update_disk(model_convolved)

then = datetime.now()
modelfm_here = diskobj.fm_parallelized()
print('DiskFM RDI', datetime.now() - then)

return_by_fm_parallelized = modelfm_here[0]  # first KL

fits.writeto(
    dir_test+ "test_results/" + fileprefix + 'returnedbydiskFMparallelized.fits',
    return_by_fm_parallelized,
    overwrite=True,
)