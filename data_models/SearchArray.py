import os

from pymatreader import read_mat


PATH = r"S:\Lab-Shared\Experiments\LWS Free Viewing Demo\Stimuli\generated_stim1\color"
matfile = read_mat(os.path.join(PATH, "image_1.mat"))['imageInfo']

