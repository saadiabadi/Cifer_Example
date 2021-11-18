from fedn.utils.kerashelper import KerasHelper
from models.vgg import create_seed_model
import numpy as np

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model(dimension='VGG16')
	outfile_name = "seed.npz"
	weights = model.get_weights()
	helper = KerasHelper()
	helper.save_model(weights, outfile_name)
	#np.savez_compressed(outfile_name, **model.state_dict())