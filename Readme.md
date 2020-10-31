# Introduction

This repository includes the source code of my submitted Acoustic Scene Classification (ASC) system to Task 1B of the DCASE challenge 2020. The code is based on Python 3.5 and uses PyTorch 1.0.0.

The submitted system used a CNN architecture that is found by random search in a predefined search space. The architectures in the search space have 2~4 convolutional blocks. Each block contains 1~2 efficient convolutional modules and a max/average pooling layer. The possible choices of convolutional modules include depth-wise separable convolution (DSC) module and inverted residual with linear bottleneck module. The possible kernel size for a convolutional module can be 3x3 or 5x5. Thus, the random sampled architectures from the search space should have different receptive field, depth (number of layers) and width (number of kernels in each convolutional module).



More details of this work can be found in [here](http://dcase.community/documents/workshop2020/proceedings/DCASE2020Workshop_Wu_18.pdf). 

# How to Use

There are two steps to run the system. 

1. Feature extraction, i.e., extract the average-difference time-frequency representation from the binaural audio waveforms. The CNNs will be trained with these extracted features.
2. Model searching. Repeatedly sampling model architectures from the search space, train and fast evaluate the sampled architecture, and save its accuracy in development dataset for further analysis.

The evaluated models will be recorded in the csv file "found_models_list.csv" for further analysis.


## Pre-requisite:
The code is based on Python 3.5 and uses PyTorch 1.0.0. The libraries' versions for running the code are listed below. However, the code should be able to run with libraries of newer versions. 

- numpy.__version__=='1.14.0'
- soundfile.__version__=='0.9.0'
- yaml.__version__=='3.12'
- cv2.__version__=='3.4.2'
- scipy.__version__=='1.0.0'
- imageio.__version__=='2.5.0'
- pickle.__version__=='$Revision: 72223 $'
- sklearn.__version__=='0.18.2'
- matplotlib.__version__=='2.0.2'



## Audio Feature Extraction

The code are in the "Step1_Feature_extraction" folder.

To do feature extraction, execute the python program "extr-asc.py". Before running the program, set the paths of the raw dataset and the output folders. 'raw_data_folder' is the path to audios in development dataset. 'output_feature_folder' is where extracted features are stored. 'spectrograms_folder' includes the feature images for visualization purpose only.

```python
config = { ...
	'raw_data_folder': '.../TAU-urban-acoustic-scenes-2020-3class-development/audio',
	'output_feature_folder': '.../features/development/scalogram-avgDiff',
	'spectrograms_folder': '.../features/development/scalogram-avgDiff-imgs',
	...
	}
```
Then run the script by
```python
python extr-asc.py
```

## Search and Evaluate Model Architectures

The codes are in the "Step2_Search_models" folder. 

Before running the script, check and modify the configurations in "fast_train.py":

```python
#===============================
# Configurations
#===============================
	data_config = {	'feat_key': 'feat_scg_avgDiff',
					'file_extension': 'scalogram',
					...
				} 
```
Then run the script "run.sh".

The evaluated models will be recorded in this csv file: "found_models\found_models_list.csv".