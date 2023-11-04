# FaceDiffuser 
Code repository for the implementation of: *FaceDiffuser: Speech-Driven Facial Animation Synthesis Using Diffusion.*
> 
> This GitHub repository contains PyTorch implementation of the work presented above. 
> FaceDiffuser generates facial animations based on raw audio input of speech sequences. 
> By employing the diffusion mechanism our model produces different results for every new
> inference.

>We reccomend visiting the project [website](https://uuembodiedsocialai.github.io/FaceDiffuser/) and watching the supplementary video.

## Environment

- Linux and Windows (tested on Windows 10 and 11)
- Python 3.9+
- PyTorch 1.10.1+cu111

## Dependencies

- [ffmpeg](https://www.ffmpeg.org/download.html)
- Check the required python packages and libraries in `requirements.txt`.
- Install them by running the command: `pip install -r requirements.txt`

## Data
### BIWI

The [Biwi 3D Audiovisual Corpus of Affective Communication](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) dataset is available upon request for research or academic purposes. 


#### BIWI Data Preparation and Data Pre-process 
In the interest of fair comparison with previous works, BIWI dataset was prepared according to the data processing that was done in [CodeTalker](https://github.com/Doubiiu/CodeTalker). Please follow this [link](https://github.com/Doubiiu/CodeTalker/tree/main/BIWI) and follow the instructions there to prepare the dataset. After processing, the `*.npy` files should be in `data/BIWI/vertices_npy/` folder whereas the `.wav` files should be in `data/BIWI/wav/` folder. This processing only prepares the emotional subset sequences. The results reported in the paper are based on this pre-processed data. 

P.S.: [FaceXHuBERT](https://github.com/galib360/FaceXHu) also provides a data processing workflow that processes the full BIWI dataset (including neutral and emotional sequences). 

### VOCASET

Download the training data from: https://voca.is.tue.mpg.de/download.php.

Place the downloaded files `data_verts.npy`, `raw_audio_fixed.pkl`, `templates.pkl` and `subj_seq_to_idx.pkl` in the folder `data/vocaset/`.
Read the downloaded data and convert it to .npy and .wav format accepted by the model. Run the following instructions for this:

```commandline
cd data/vocaset
python process_voca_data.py
```

### Multiface

Download the Multiface dataset by following the instructions here: https://github.com/facebookresearch/multiface.

Keep in mind that only `mesh` and `audio` data is needed for training the model.


```commandline
cd data/mutliface
python convert_topology.py
python preprocess.py
```

### Beat

Download the Beat dataset from here: https://pantomatrix.github.io/BEAT/.
Keep in mind that only the facial motion (stored in json files)  and audio (stored in wav files) are needed for training the model.

Follow the instructions in `data/beat` for preprocessing the data before training.


## Model Training 

### Training and Testing

| Arguments     | BIWI  | VOCASET | Multiface | UUDaMM | BEAT |
|---------------|-------|---------|-----------|--------|------|
| --dataset     |  BIWI | vocaset | multiface |  damm  | beat |
| --vertice_dim | 70110 |  15069  |   18516   |   192  |  51  |
| --output_fps  |   25  |    30   |     30    |   30   |  30  |

- Train the model by running the following command:
	```
	python main.py
	```
	The test split predicted results will be saved in the `result/`. The trained models (saves the model in 25 epoch interval) will be saved in the `save/` folder.


### Predictions

- Download the trained weights from [here](https://mega.nz/folder/jlBF0Dpa#U3G1lJCZ4dijMoSc9gmqSg) and add them to the folder `pretrained_models`.
- To generate predictions use the commands:

BIWI
```commandline
python predict.py --dataset BIWI --vertice_dim 70110 --feature_dim 512 --output_fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F2 F3 F4 M3 M4 M5" --model_name "pretrained_BIWI" --fps 25 --condition "F2" --subject "F2" --diff_steps 500 --gru_dim 512 --wav_path "test.wav"  
```
Vocaset
```commandline
python predict.py --dataset vocaset --vertice_dim 15069 --feature_dim 256 --output_fps 30 --train_subjects "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA" --test_subjects "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA" --model_name "pretrained_vocaset" --fps 30 --condition "FaceTalk_170728_03272_TA" --subject "FaceTalk_170731_00024_TA" --diff_steps 1000 --gru_dim 256 --wav_path "test.wav"
```
Multiface
```commandline
python predict.py --dataset multiface --vertice_dim 18516 --feature_dim 256 --output_fps 30 --train_subjects "2 3 6 7 9 10 11 12 13" --test_subjects "1 4 5 8" --model_name "pretrained_multiface" --fps 30 --condition "2" --subject "1" --diff_steps 1000 --gru_dim 256 --wav_path "test.wav"
```
    
### Visualization

- Run the following command to render the predicted test sequences stored in `result/`:

	```
	python render_result.py
	```
	The rendered videos will be saved in the `renders/videos/` folder.

### Trained Weights
The trained weights can be downloaded from [THIS](https://mega.nz/folder/jlBF0Dpa#U3G1lJCZ4dijMoSc9gmqSg) link. 

### Acknowledgements

We borrow and adapt the code from [FaceXHuBERT](https://github.com/galib360/FaceXHuBERT), 
[MDM](https://github.com/GuyTevet/motion-diffusion-model), [EDGE](https://edge-dance.github.io/), [CodeTalker](https://github.com/Doubiiu/CodeTalker).
Thanks for making their code available and facilitating future research.
Additional thanks to [huggingface-transformers](https://huggingface.co/) for the implementation of HuBERT.

We are also grateful for the publicly available datasets used during this project:
- ETHZ-CVL for providing the B3D(AC)2 dataset
- MPI-IS for releasing the VOCASET dataset. 
- Facebook Research for realising the Multiface dataset.
- Utrecht University for the UUDaMM dataset.
- The authors of the BEAT dataset.

Any third-party packages are owned by their respective authors and must be used under their respective licenses.
