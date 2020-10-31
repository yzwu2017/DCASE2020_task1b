import sys
import numpy as np
import glob #use to get file list in a folder
import soundfile as sf
import librosa #use to extract MFCC feature
import yaml #use to save and read statistics
import matplotlib.pyplot as plt
import scipy.misc
import scipy.signal
import cv2
import pdb

import time
from multiprocessing import Pool
from scipy import ndimage

from src.funcs import * 

import time
import imageio


def convert_to_uint8(image):
	output_image = (image - image.min())/(image.max() - image.min()) * 255
	output_image = output_image.astype(np.uint8)
	return output_image


class Extract_Feature_Batch:
	def __init__(self, config):
		self.cfg = config
	def extract(self,audio_list):
		win_length = int(self.cfg['win_length_in_seconds'] * self.cfg['SR'])
		hop_length = int(self.cfg['hop_length_in_seconds'] * self.cfg['SR'])
		count = 0
		for file_id, audio_file_path in enumerate(audio_list):
			start_time = time.time()
			current_feature_file = get_feature_filename(audio_file_path, self.cfg['output_feature_folder'], extension=self.cfg['feature_type'])
			if not os.path.isfile(current_feature_file) or self.cfg['overwrite']:
				# Load audio data
				if os.path.isfile(audio_file_path):
					data, samplerate = sf.read(audio_file_path)
				else:
					raise IOError("Audio file not found [%s]" % os.path.split(audio_file_path)[1])
				#=================================
				# Extract features
				#=================================
				if self.cfg['feature_type'] == 'logmel':
					data_left = data[:,0]
					data_right = data[:,1]
					logmel_left = extract_logmel(data=data_left, sr=self.cfg['SR'], win_length=win_length, hop_length=hop_length, config=self.cfg)
					logmel_right = extract_logmel(data=data_right, sr=self.cfg['SR'], win_length=win_length, hop_length=hop_length, config=self.cfg)
					logmel_mid = (logmel_left + logmel_right) / 2.0
					logmel_diff = logmel_left - logmel_right
					logmel_avgDiff = np.array([logmel_mid,logmel_diff])

					feature_data = {'feat_avgDiff': logmel_avgDiff}
					# Save feature data
					pickle.dump(feature_data, open(current_feature_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)		
					if self.cfg['save_spectrograms']:
						img_file_name = current_feature_file.split('/')[-1]
						img_file_name = img_file_name.split('.')[0]

						specgram_img = logmel_mid.T
						specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
						specgram_img = 255 * (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
						specgram_img = specgram_img.astype(np.uint8)
						imageio.imwrite(self.cfg['spectrograms_folder'] + '/' + img_file_name +'-avg.jpg', specgram_img)

						specgram_img = logmel_avgDiff.T
						specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
						specgram_img = 255 * (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
						specgram_img = specgram_img.astype(np.uint8)
						imageio.imwrite(self.cfg['spectrograms_folder'] + '/' + img_file_name +'-diff.jpg', specgram_img)

				if self.cfg['feature_type'] == 'scalogram':
					data_left = data[:,0]
					data_right = data[:,1]
					scalogram_left = extract_scalogram(data=data_left, win_length=win_length, hop_length=hop_length, config=self.cfg)
					scalogram_right = extract_scalogram(data=data_right, win_length=win_length, hop_length=hop_length, config=self.cfg)
					scalogram_mid = (scalogram_left + scalogram_right) / 2.0
					scalogram_diff = scalogram_left - scalogram_right
					scalogram_avgDiff = np.array([scalogram_mid,scalogram_diff])
					feature_data = {'feat_scg_avgDiff': scalogram_avgDiff}
					# Save feature data
					pickle.dump(feature_data, open(current_feature_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)		
					if self.cfg['save_spectrograms']:
						img_file_name = current_feature_file.split('/')[-1]
						img_file_name = img_file_name.split('.')[0]

						specgram_img = scalogram_mid.T
						specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
						specgram_img = 255 * (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
						specgram_img = specgram_img.astype(np.uint8)
						imageio.imwrite(self.cfg['spectrograms_folder'] + '/' + img_file_name +'-avg.jpg', specgram_img)

						specgram_img = scalogram_diff.T
						specgram_img = np.flip(specgram_img,axis=0) #Making the bottom of the image be the low-frequency part.
						specgram_img = 255 * (specgram_img - np.min(specgram_img)) / (np.max(specgram_img) - np.min(specgram_img))
						specgram_img = specgram_img.astype(np.uint8)
						imageio.imwrite(self.cfg['spectrograms_folder'] + '/' + img_file_name +'-diff.jpg', specgram_img)


			count = count + 1
			elapsed = time.time() - start_time
			print("[Time: %.2fs] Progress %.1f%% | " % (elapsed,(file_id+1) / float(len(audio_list)) * 100) +  os.path.split(audio_file_path)[1] + "                            '\r'")
		return 


N_CORES = 4
config = {	'save_spectrograms': True, # If True, the extracted features will be saved as images for visualization purpose.
			'overwrite': False,  # Overwrite flag: Whether overwritting the existing feature file or not.
			'raw_data_folder': '/.../TAU-urban-acoustic-scenes-2020-3class-development/audio',
			'output_feature_folder': '/.../features/DCASE2020_task1b/development/scalogram-J10Q16-avgDiff',
			'spectrograms_folder': '/.../features/DCASE2020_task1b/development/scalogram-J10Q16-avgDiff-imgs',
			'feature_type': 'scalogram',
			'SR': 48000,                      # The sampling frequency for feature extraction.
			'win_length_in_seconds': 0.025,   # the window length (in second). Default: 0.025
			'hop_length_in_seconds': 0.010,   # the hop length (in second). Default: 0.010
			'window': 'hamming_asymmetric',   # [hann_asymmetric, hamming_asymmetric]
			'n_fft': 2048,                    # FFT length       
			'n_mels': 128,                     # Number of MEL bands used
			'fmin': 20,                       # Minimum frequency when constructing MEL bands
			'fmax': 24000,                    # Maximum frequency when constructing MEL band
			'N_delta': 4,
			'J': 10, # The parameter `J` specifies the maximum scale of the filters as a power of two. In other words, the largest filter will be concentrated in a time interval of size `2**J`.
			'Q': 16, # The `Q` parameter controls the number of wavelets per octave in the first-order filter bank.
			}

# config_file = 'feature_config.yaml'
# with open(config_file,'r') as f:
# 	config = yaml.load(f)

#=======================
# Setting for feature extractions
#=======================

raw_audio_list = glob.glob(config['raw_data_folder'] + '/*.wav')
n_audio = len(raw_audio_list)

# split the whole audio list into sub-lists.
n_audio_split = int(np.ceil(n_audio / float(N_CORES)))
sub_lists = []
n_remains = n_audio
current_idx = 0
for i in range(N_CORES):
	if n_audio >= n_audio_split:
		sublist = raw_audio_list[current_idx : current_idx+n_audio_split]
		sub_lists.append(sublist)
		n_remains = n_remains - n_audio_split
		current_idx = current_idx+n_audio_split
	else:
		sublist = raw_audio_list[current_idx:]
		sub_lists.append(sublist)

ensure_folder_exists(config['output_feature_folder'])
if config['save_spectrograms']:
	ensure_folder_exists(config['spectrograms_folder'])

# Save feature configuration
with open(config['output_feature_folder'] + '/feature.config','w') as yaml_file:
	yaml.dump(config, yaml_file, default_flow_style=False)	


extract_feature_batch = Extract_Feature_Batch(config)

#extract_feature_batch.extract(raw_audio_list) # Use only one thread for feature extraction.
#========================
# Start Feature Extraction Using Multiple Cores.
#========================
#mark the start time
startTime = time.time()
#create a process Pool with N_CORES processes
pool = Pool(processes=N_CORES)
# map doWork to availble Pool processes
pool.map(extract_feature_batch.extract, sub_lists)
#mark the end time
endTime = time.time()
#calculate the total time it took to complete the work
workTime =  endTime - startTime
#print results
print("The job took " + str(workTime) + " seconds to complete")
