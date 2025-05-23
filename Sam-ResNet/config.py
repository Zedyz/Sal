#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# version (0 for SAM-VGG and 1 for SAM-ResNet)
version = 1
# batch size
b_s = 1
# number of rows of input images
shape_r = 240
# number of cols of input images
shape_c = 320
# number of rows of downsampled maps
shape_r_gt = 30
# number of cols of downsampled maps
shape_c_gt = 40
# number of rows of model outputs
shape_r_out = 480
# number of cols of model outputs
shape_c_out = 640
# final upsampling factor
upsampling_factor = 16
# number of epochs
nb_epoch = 40
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training images
imgs_train_path = 'train_orig'
# path of training maps
maps_train_path = 'train_fdm'
# path of training fixation maps
fixs_train_path = 'train_fix'
# number of training images
nb_imgs_train = 392
# path of validation images
imgs_val_path = 'test_orig'
# path of validation maps
maps_val_path = 'test_fdm'
# path of validation fixation maps
fixs_val_path = 'test_fix'
# number of validation images
nb_imgs_val = 58