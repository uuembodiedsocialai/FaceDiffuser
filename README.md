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

The [Biwi 3D Audiovisual Corpus of Affective Communication](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) dataset is available upon request for research or academic purposes. You will need the following files from the the dataset: 

- faces01.tgz, faces02.tgz, faces03.tgz, faces04.tgz, faces05.tgz and rest.tgz
- Place all the faces0*.tgz archives in `data/BIWI/ForProcessing/FaceData/` folder
- Place the rest.tgz archive in `data/BIWI/ForProcessing/rest/` folder


#### Data Preparation and Data Pre-process 
Follow the steps below sequentially as they appear - 

- You will need [Matlab](https://mathworks.com/products/matlab.html) installed on you machine to prepapre the data for pre-processing
- Open Anaconda Promt CLI, activate FaceXHuBERT env in the directory- `data/BIWI/ForPorcessing/rest/`
- Run the following command
    ```
    tar -xvzf rest.tgz
    ```
- After extracting, you will see the `audio/` folder that contains the input audios needed for network training in .wav format
- Run the `wav_process.py` script. This will process the `audio/` folder and copy the needed audio sequences with proper names to `data/BIWI/wav/` folder for training
    ```
    python wav_process.py
    ```
- Open Anaconda Promt CLI, activate FaceXHuBERT env in the directory- `BIWI/ForPorcessing/FaceData/`
- Run the following command for extracting all the archives. Replace `*` with (1-5 for five archives)
    ```
    tar -xvzf faces0*.tgz
    ``` 
- After extracting, you will see a folder named `faces/`. Move all the .obj files from this folder  (i.e. F1.obj-M6.obj) to `FaceXHuBERT/BIWI/templates/` folder 
- Run the shell script `Extract_all.sh`. This will extract all the archives for all subjects and for all sequences. You will have frame-by-frame vertex data in `frame_*.vl` binary file format  
- Run the Matlab script `vl2csv_recusive.m`. This will convert all the `.vl` files into `.csv` files
- Run the `vertex_process.py` script. This will process the data and place the processed data in `FaceXHuBERT/BIWI/vertices_npy/` folder for network training
    ```
    python vertex_process.py
    ```

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

### Visualization

- Run the following command to render the predicted test sequences stored in `result/`:

	```
	python render_result.py
	```
	The rendered videos will be saved in the `renders/videos/` folder.


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
