from protosc.pipeline import BasePipeElement
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix


class FourierFeatures(BasePipeElement):
    def __init__(self, n_angular=8, n_spatial=7, cut_circle=True,
                 absolute=True):
        """Use fourier transformation on an image.

        At the moment reverse transformation/visualization is half implemented.
        The results are symmetrized: surfaces on opposite sides of the middle
        are averaged.

        Arguments
        ---------
        n_angular: int
            The number of angular steps in the coarse graining.
        n_spatial: int
            The number ofradial steps in the coarse graining.
        cut_circle: bool
            Whether only the inner circle has data (preprocessing).
        absolute: bool
            Whether to take the absolute values before coarse graining.

        Returns
        -------
        X: np.ndarray
            Feature matrix. If cut_circle is true, then the dimensions are
            n_absolute*n_spatial, otherwise it will be slightly larger.
        """
        self.n_angular = n_angular
        self.n_spatial = n_spatial
        self.cut_circle = cut_circle
        self.absolute = absolute

    def _execute(self, img):
        return fourier_features(
            img, n_angular=self.n_angular, n_spatial=self.n_spatial,
            cut_circle=self.cut_circle, absolute=self.absolute)

    @property
    def name(self):
        name = super(FourierFeatures, self).name
        name += f"_a{self.n_angular}s{self.n_spatial}c{self.cut_circle}"
        name += f"ab{self.absolute}"
        return name


class AbsoluteFeatures(BasePipeElement):
    def _execute(self, features):
        return np.absolute(features)


def transform_matrix(shape, n_angular=8, n_spatial=7, return_inverse=True,
                     return_ids=False, cut_circle=True):
    # Compute the x and y values for all pixels from the middle.
    size = shape[0]*shape[1]
    X, Y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    middle = np.array([shape[0]//2, shape[1]//2])
    X -= middle[0]
    Y -= middle[1]

    # Compute the radius and angle for each pixel.
    radius = np.sqrt(X**2 + Y**2)
    angle = np.arctan2(Y, X)

    # Compute the coarse graining for each pixel.
    d_angle = 2*np.pi/n_angular
    d_radius = np.min(middle)/n_spatial
    angle_id = ((2*angle/d_angle + 0.5*(2*n_angular+1)
                 ) % (2*n_angular)).astype(int)
    angle_id = angle_id % n_angular
    radius_id = (radius/d_radius).astype(int)
    all_id = angle_id+radius_id*n_angular
    unique_id = np.unique(all_id)
    if all_id.max() > len(unique_id)-1:
        conversion = np.zeros(all_id.max()+1, dtype=int)
        conversion[unique_id] = np.arange(len(unique_id))
        all_id = conversion[all_id]

#     Set up the sparse matrix that transforms image data.
    indptr = np.arange(size+1)
    indices = all_id.reshape(-1)
    data = np.ones(size, dtype=int)
    trans_shape = (all_id.max()+1, size)

    # If there are no values outside the inner circle:
    if cut_circle:
        circle_mask = (radius_id.reshape(-1) < n_spatial)
        trans_shape = (all_id.reshape(-1)[circle_mask].max()+1, size)

        # Remove ids outside the inner circle.
        all_id[radius_id >= n_spatial] = -1
        indptr = np.append([0], np.cumsum(circle_mask))
        indices = indices[circle_mask]
        data = data[circle_mask]

    # Create transformation matrix
    trans_matrix = csc_matrix((data, indices, indptr),
                              shape=trans_shape)
    results = []

    results.append(trans_matrix)
    # Return the coarse grained ids for all pixels.
    if return_ids:
        results.append(all_id.reshape(-1))

    if not return_inverse:
        if len(results) == 1:
            return trans_matrix
        return results

    # Compute the inverse matrix
    # Count the number of pixels for each used cell.
    idx, temp_counts = np.unique(all_id[all_id != -1], return_counts=True)
    counts = np.zeros(all_id.max()+1)
    counts[idx] = temp_counts
    indptr = np.arange(size+1)
    indices = all_id.reshape(-1)
    data = 1/counts[all_id.reshape(-1)]
    index_mask = (indices >= 0)
    indptr = np.cumsum(np.append([False], index_mask))
    data = data[index_mask]
    indices = indices[index_mask]

    # Create sparse matrix.
    inv_trans_matrix = csr_matrix((data, indices, indptr),
                                  shape=(trans_shape[1], trans_shape[0]))
    results.append(inv_trans_matrix)
    return results


def fourier_features(img, *args, absolute=True, **kwargs):
    fft_map = np.fft.fftshift(
        np.fft.fft2(img-np.mean(img, axis=(0, 1)), axes=(0, 1)))
    if absolute:
        fft_map = np.absolute(fft_map)
    trans = transform_matrix(fft_map.shape, *args, return_inverse=False,
                             **kwargs)
    return trans.dot(fft_map.reshape(-1, fft_map.shape[2]))


class HOGFeatures(BasePipeElement):
        """Extract HOG feature from an image.

        Arguments
        ---------
        orientations: int
            The number of orientation bins
        HOG_cellsize: [int,int]
            The size of the (non-overlapping) cells

        Returns
        -------
        HOGs: vector of HOG feature values
        refGrid_HOG: matrix where each value corresponds to an index in HOGs. 
        Use this to find where in the image a particular HOG feature value comes from
        """
    def __init__(self, orientations=9, HOG_cellsize=[10,10]):
        self.orientations = orientations
        self.HOG_cellsize = HOG_cellsize

    def _execute(self, img):
        return HOG_features(
            img, orientations=self.orientations, HOG_cellsize=self.HOG_cellsize)
        
def HOG_features(img,orientations,HOG_cellsize):
    from skimage.feature import hog
    import numpy as np
    HOGs            = hog(img, orientations, HOG_cellsize,cells_per_block=(1, 1),visualize=False,multichannel=True)
    refGrid_HOG     = np.zeros([np.int(np.floor(img.shape[0]/HOG_cellsize[0])),np.int(np.floor(img.shape[1]/HOG_cellsize[1])),orientations])
    c               = 0;    
    for x in range(0,refGrid_HOG.shape[1]):
        for y in range(0,refGrid_HOG.shape[0]):
            for z in range(0,refGrid_HOG.shape[2]):
                refGrid_HOG[y,x,z] = c
                c = c+1
    return HOGs,refGrid_HOG 

class ColorFeatures(BasePipeElement):
        """Extract Color distribution features from image

        Arguments
        ---------
        nsteps: int
            The number of bins used on the pdf of color values


        Returns
        -------
        ColorDistributions: vector of color pdf values
        refGrid: matrix where each value corresponds to an index in ColorDistributions. 
        Use this to find where in the image a particular feature value comes from
        """
    def __init__(self, nsteps=25):
        self.nsteps = nsteps

    def _execute(self, img):
        return Color_features(
            img, nsteps=self.nsteps)
        
def Color_features(img,nsteps):
    import numpy as np

    #preallocate h
    ColorDistributions = []
  
    refGrid = np.zeros([img.shape[2],nsteps])
    count   = 0    
    for l in range(0,img.shape[2]):
        count = count+1
        ColorDistributions_temp,b   = np.histogram(np.reshape(img[:,:,l],img.shape[0]*img.shape[1]),nsteps,density=True) 
        ColorDistributions          = np.concatenate((ColorDistributions,ColorDistributions_temp))
        refGrid[count-1,:]          = np.array(range(nsteps*(count-1),nsteps*(count)))
    return ColorDistributions, refGrid 

class PixelFeatures(BasePipeElement):
        """Extract pixel intesity features from image

        Arguments
        ---------
        newsize: [int,int]
            prior to extracting the pixel intensities, the image is converted to this size to reduce the number of features


        Returns
        -------
        Pixel_Intensities: vector of pixel intensities
        refGrid: matrix where each value corresponds to an index in Pixel_Intensities. 
        Use this to find where in the image a particular feature value comes from
        """
    def __init__(self, newsize=[25,25]):
        self.newsize = newsize

    def _execute(self, img):
        return Pixel_features(
            img, newsize=self.newsize)
        
def Pixel_features(img,newsize):
    import numpy as np
    from skimage.transform import resize
    
    img = resize(img,newsize)
    Pixel_Intensities           = np.reshape(img,[1,img.shape[0]*img.shape[1],img.shape[2]])
    refGrid_Pixel_Intensities   = np.zeros([img.shape[0],img.shape[1],img.shape[2]])
    
    c = 0;    
    for x in range(0,img.shape[1]-1):
        for y in range(0,img.shape[0]-1):
            for z in range(0,img.shape[2]-1):
                refGrid_Pixel_Intensities[y,x,z]=c
                c = c+1;
                
    return Pixel_Intensities,refGrid_Pixel_Intensities                 

class SetColorChannels(BasePipeElement):
        """Image preprocessing step for color conversion and selecting specific color channels
        

        Arguments
        ---------
        convert2cielab: bol
            use to convert rgb to cielab (convert2cielab = True/Flase)
        get_layers: array
            Select which channels of the image to keep (get_layers= [0,1,2])

        Returns
        -------
        img: the adjusted image
        
        """
    def __init__(self, convert2cielab=False,get_layers=[]):
        self.convert2cielab = convert2cielab
        self.get_layers = get_layers

    def _execute(self, img):
        return set_color_channels(
            img, convert2cielab=self.convert2cielab,get_layers=self.get_layers)
        
def set_color_channels(img,convert2cielab,get_layers): 
    # preprocessing step for images
    # use to convert rgb to cielab (convert2cielab = True/Flase)
    # Select which channels of the image to keep (get_layers= [0,1,2])
    
    import skimage as sk
                
    # Convert RGB to Cie Lab
    if convert2cielab:
        img = sk.color.rgb2lab(img)
    # Check which image layers to include    
    if get_layers==[]:
        get_layers = range(0,img.shape[2])
        
    newimg = img[:,:,get_layers] 
    
    return newimg

