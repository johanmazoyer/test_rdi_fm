import numpy as np
import astropy.io.fits as fits

import pyklip.rdi as rdi
import glob
import pyklip.instruments.GPI as GPI
import pyklip.parallelized as parallelized

dir_test = '/Users/jmazoyer/Desktop/GPI_librairy_RDI_dataset/'

aligned_center = [140,140]


lib_files = sorted(glob.glob(dir_test+'*.fits'))
#print data_files
datasetlib = GPI.GPIData(lib_files, highpass=False)
datasetlib.spectral_collapse(collapse_channels=1,align_frames=True)

toto = datasetlib.input
toto_nanpix = np.where(np.isnan(toto))
toto[toto_nanpix] = 0
datasetlib.input = toto


# make the PSF library
# we need to compute the correlation matrix of all images vs each other since we haven't computed it before
psflib = rdi.PSFLibrary(datasetlib.input,aligned_center , datasetlib.filenames, compute_correlation=True)

# save the correlation matrix to disk so that we also don't need to recomptue this ever again
# In the future we can just pass in the correlation matrix into the PSFLibrary object rather than having it compute it
psflib.save_correlation(dir_test+"test_results/corr_matrix-SMALLTEST.fits", clobber=True)


# read in the correlation matrix we already saved
corr_matrix = fits.getdata(dir_test+"test_results/corr_matrix-SMALLTEST.fits")



# make the PSF library again, this time we have the correlation matrix
psflib = rdi.PSFLibrary(datasetlib.input, aligned_center, datasetlib.filenames, correlation_matrix=corr_matrix)


data_files = sorted(glob.glob(dir_test + 'S20160318S*.fits'))
#print data_files
dataset = GPI.GPIData(data_files, highpass=False)
dataset.spectral_collapse(collapse_channels=1,align_frames=True)


toto = dataset.input
toto_nanpix = np.where(np.isnan(toto))
toto[toto_nanpix] = 0
dataset.input = toto


psflib.prepare_library(dataset)



# now we can run RDI klip
# as all RDI images are aligned to aligned_center, we need to pass in that aligned_center into KLIP
numbasis=[1] # number of KL basis vectors to use to model the PSF. We will try several different ones
maxnumbasis=2 # maximum number of most correlated PSFs to do PCA reconstruction with
annuli=1
subsections=1 # break each annulus into 4 sectors
parallelized.klip_dataset(dataset, outputdir=dir_test + "test_results/", fileprefix="pyklip_rditest", annuli=annuli,
                        subsections=subsections, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="RDI",
                        aligned_center=aligned_center, highpass=False, psf_library=psflib)

parallelized.klip_dataset(dataset, outputdir=dir_test + "test_results/", fileprefix="pyklip_aditest", annuli=annuli,
                        subsections=subsections, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="ADI",
                        aligned_center=aligned_center, highpass=False)
