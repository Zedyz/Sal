from __future__ import division
import os, cv2, sys
import numpy as np

# Use TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Local imports (make sure these are updated too, if needed)
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
from models import sam_vgg, sam_resnet, kl_divergence, correlation_coefficient, nss


def generator(b_s, phase_gen='train_orig'):
    """
    Generator function yielding ([images, gaussian], [map, map, fix]),
    where images come from disk, and 'gaussian' is a dummy prior input.
    """
    if phase_gen == 'train_orig':
        images = [os.path.join(imgs_train_path, f)
                  for f in os.listdir(imgs_train_path)
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [os.path.join(maps_train_path, f)
                for f in os.listdir(maps_train_path)
                if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [os.path.join(fixs_train_path, f)
                for f in os.listdir(fixs_train_path)
                if f.endswith('.mat')]
    elif phase_gen == 'val':
        images = [os.path.join(imgs_val_path, f)
                  for f in os.listdir(imgs_val_path)
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
        maps = [os.path.join(maps_val_path, f)
                for f in os.listdir(maps_val_path)
                if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs = [os.path.join(fixs_val_path, f)
                for f in os.listdir(fixs_val_path)
                if f.endswith('.mat')]
    else:
        raise NotImplementedError(f"Unknown phase_gen={phase_gen}")

    images.sort()
    maps.sort()
    fixs.sort()

    # Dummy Gaussian prior input of shape (b_s, nb_gaussian, shape_r_gt, shape_c_gt)
    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        # Load a batch
        Y = preprocess_maps(maps[counter:counter + b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], shape_r_out, shape_c_out)
        X_imgs = preprocess_images(images[counter:counter + b_s], shape_r, shape_c)

        # yield inputs and outputs
        yield [X_imgs, gaussian], [Y, Y, Y_fix]

        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [os.path.join(imgs_test_path, f)
              for f in os.listdir(imgs_test_path)
              if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))
    counter = 0
    while True:
        X_imgs = preprocess_images(images[counter:counter + b_s], shape_r, shape_c)
        # Instead of: yield [X_imgs, gaussian]
        # Do:
        yield ([X_imgs, gaussian], None)

        counter = (counter + b_s) % len(images)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise NotImplementedError("Usage: python main.py <phase> [test_image_path]")
    else:
        phase = sys.argv[1]

        # Define inputs
        x = Input((3, shape_r, shape_c))  # channels-first shape
        x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))

        # Build model
        if version == 0:
            print("Building SAM-VGG model...")
            outputs = sam_vgg([x, x_maps])  # [outs_up, outs_up, outs_up]
            m = Model(inputs=[x, x_maps], outputs=outputs)
            print("Compiling SAM-VGG...")
            m.compile(
                optimizer=RMSprop(learning_rate=1e-4),
                loss=[kl_divergence, correlation_coefficient, nss]
            )
        elif version == 1:
            print("Building SAM-ResNet model...")
            outputs = sam_resnet([x, x_maps])
            m = Model(inputs=[x, x_maps], outputs=outputs)
            print("Compiling SAM-ResNet...")
            m.compile(
                optimizer=RMSprop(learning_rate=1e-4),
                loss=[kl_divergence, correlation_coefficient, nss]
            )
        else:
            raise NotImplementedError("Unknown version. Must be 0 or 1.")

        if phase == 'train_orig':
            # Check batch-size multiples
            if (nb_imgs_train % b_s != 0) or (nb_imgs_val % b_s != 0):
                print("Number of training/validation images must be multiple of batch size.")
                sys.exit(1)

            steps_per_epoch = nb_imgs_train // b_s
            validation_steps = nb_imgs_val // b_s

            if version == 0:
                print("Training SAM-VGG...")
                m.fit(
                    generator(b_s=b_s, phase_gen='train_orig'),
                    steps_per_epoch=steps_per_epoch,
                    epochs=nb_epoch,  # modern Keras arg is 'epochs'
                    validation_data=generator(b_s=b_s, phase_gen='val'),
                    validation_steps=validation_steps,
                    callbacks=[
                        EarlyStopping(patience=30),
                        ModelCheckpoint('weights.sam-vgg.{epoch:02d}-{val_loss:.4f}.h5',
                                        save_best_only=True,
                                        monitor='val_loss')
                    ]
                )
            else:  # version == 1
                print("Training SAM-ResNet...")
                m.fit(
                    generator(b_s=b_s, phase_gen='train_orig'),
                    steps_per_epoch=steps_per_epoch,
                    epochs=nb_epoch,
                    validation_data=generator(b_s=b_s, phase_gen='val'),
                    validation_steps=validation_steps,
                    callbacks=[
                        EarlyStopping(patience=30),
                        ModelCheckpoint('weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.h5',
                                        save_best_only=True,
                                        monitor='val_loss')
                    ]
                )

        elif phase == "test_orig":
            # Usage: python main.py test_orig <imgs_test_path>
            if len(sys.argv) < 3:
                raise SyntaxError(
                    "You must provide path to test images, e.g. python main.py test_orig /path/to/test_imgs/")
            imgs_test_path = sys.argv[2]
            output_folder = 'predictions/'
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            file_names = [f for f in os.listdir(imgs_test_path)
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            file_names.sort()
            nb_imgs_test = len(file_names)

            if nb_imgs_test % b_s != 0:
                print("Number of test images must be a multiple of batch size. Adjust b_s or # of test images.")
                sys.exit(1)

            # Load pretrained weights
            if version == 0:
                print("Loading SAM-VGG weights...")
                m.load_weights('weights.sam-resnet.15-3.1824.h5')
            elif version == 1:
                print("Loading SAM-ResNet weights...")
                m.load_weights('weights.sam-resnet.15-3.1824.h5')

            print(f"Predicting saliency maps for {imgs_test_path}...")
            steps_test = nb_imgs_test // b_s
            predictions_all = m.predict(
                generator_test(b_s=b_s, imgs_test_path=imgs_test_path),
                steps=steps_test
            )
            # predictions_all is a list of 3 outputs: [outs_up, outs_up, outs_up].
            # Usually the relevant saliency is predictions_all[0], shape => (nb_imgs_test, 1, shape_r_out*factor, shape_c_out*factor)

            predictions = predictions_all[0]  # take the first

            # Save each predicted map
            idx = 0
            for name in file_names:
                original_image = cv2.imread(os.path.join(imgs_test_path, name), 0)
                pred_map = predictions[idx, 0, :, :]  # shape => (H*factor, W*factor)
                res = postprocess_predictions(pred_map, original_image.shape[0], original_image.shape[1])
                cv2.imwrite(os.path.join(output_folder, name), res.astype(np.uint8))
                idx += 1

        else:
            raise NotImplementedError(f"Unknown phase={phase}")
