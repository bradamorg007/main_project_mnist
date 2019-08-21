from autoencoder_classes.old_code.variation_AutoEncoder import VariationAutoEncoder
import os

class VisualSystem:

    def __init__(self, img_shape, latent_dimensions, batch_size, RE_delta):

        self.model_dir  = '../autoencoder_classes/models/'
        self.model_folder = 'vae_test/'
        self.RE_delta = RE_delta

        self.model = VariationAutoEncoder(img_shape=img_shape,
                                          latent_space_dims=latent_dimensions,
                                          batch_size=batch_size)

        self.model.load_weights(full_path=os.path.join(self.model_dir, self.model_folder))


    def is_familular(self, sample):

        RE = self.model.model.evaluate(sample, sample, verbose=0)

        if RE > self.model.reconstruction_error + self.RE_delta:
            return False
        else:
            return True

if __name__ == '__main__':
    v = VisualSystem((28,28, 1), 3, 64, RE_delta=0.2)
    v.model.data_prep(keep_labels=[0])
    sample = v.model.x_train[80]
    label = v.model.y_train[80]

    is_familular = v.is_familular(sample)
    a = 0
