from typing import Any

import argparse
import cv2
import os
import torch
import pickle
import pyrender
import librosa
import trimesh
import ffmpeg
import gc
import numpy as np
from cog import BasePredictor, Input, Path
from models import FaceDiff
from transformers import Wav2Vec2Processor

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

os.environ['DISPLAY'] = ':1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def read_args(dataset="BIWI"):
    args = {
        "BIWI":{
            'vertice_dim': 70110,
            'feature_dim': 512,
            'device': "cuda",
            'input_fps': 50,
            'output_fps': 25,
            'diff_steps': 500,
            'device_idx': 0,
            'train_subjects': "F2 F3 F4 M3 M4 M5"
        },
        "multiface": {
            'vertice_dim': 18516,
            'feature_dim': 256,
            'device': "cuda",
            'input_fps': 50,
            'output_fps': 30,
            'diff_steps': 1000,
            'device_idx': 0
        },
        "vocaset": {
            'vertice_dim': 15069,
            'feature_dim': 256,
            'device': "cuda",
            'input_fps': 50,
            'output_fps': 30,
            'diff_steps': 1000,
            'device_idx': 0
        }
    }
    return Namespace(**args[dataset])


def create_gaussian_diffusion(args):
    # default params
    sigma_small = True
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args['diff_steps']
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule("cosine", steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=0,
        lambda_rcxyz=0,
        lambda_fc=0,
    )


def render_animation(prediction, audio_path):
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    r = pyrender.OffscreenRenderer(640, 480)

    video_woA_path = "tmp/tmp.mp4"
    video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))

    # Load the NPY file for the original frames
    frames = prediction.reshape((-1, 70110 // 3, 3))

    ref_mesh = trimesh.load_mesh('BIWI/BIWI_topology.obj', process=False)
    for i, frame in enumerate(frames):
        ref_mesh.vertices = frames[i, :, :]
        py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)

        scene = pyrender.Scene(bg_color=[25.0 / 255.0, 28.0 / 255.0, 38.0 / 255.0],
                               ambient_light=[0.02, 0.02, 0.02])
        node = pyrender.Node(
            mesh=py_mesh,
            translation=[0, 0, 0]
        )
        scene.add_node(node)

        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, _ = r.render(scene)

        output_frame = f"tmp/{i:04d}.png"

        cv2.imwrite(output_frame, color)
        frame = cv2.imread(output_frame)
        video.write(frame)

    video.release()
    video_filename = "videos/result.mp4"
    video_frames = ffmpeg.input(f"tmp/tmp.mp4")
    audio = ffmpeg.input(audio_path)

    ffmpeg.concat(video_frames, audio, v=1, a=1).output(video_filename).run(overwrite_output=True)
    del ref_mesh
    gc.collect()

    return video_filename


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        args = read_args("BIWI")
        self.model = FaceDiff(args, args.vertice_dim, cond_feature_dim=1536)
        self.model.load_state_dict(torch.load(f'pretrained_models/pretrained_BIWI.pth', map_location="cuda"))
        self.model = self.model.to(torch.device(f"cuda"))
        self.model.eval()

        self.diffusion = create_gaussian_diffusion(args)

        with open('data/BIWI/templates_scaled.pkl', 'rb') as fin:
            self.templates = pickle.load(fin, encoding='latin1')

    # Define the arguments and types the model takes as input
    def predict(self,
                audio: Path = Input(description="Speech audio file"),
                subject: str = Input(description="Subject to animate", default="F1",
                                     choices=["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
                                              "M1", "M2", "M3", "M4", "M5", "M6"]),
                conditioning_subject: str = Input(description="Conditioning Subject", default="F3",
                                                  choices=["F2", "F3", "F4", "M3", "M4", "M5"]),
                skip_timesteps: int = Input(
                    description="Number of diffussion timesteps to skip.\n 0 will give the best result but takes the longest to compute.",
                    le=500, ge=0)
                ) -> Path:
        args = read_args()
        conditioning_subjects = {
            "F2": 0,
            "F3": 1,
            "F4": 2,
            "M3": 3,
            "M4": 4,
            "M5": 5
        }
        subj = conditioning_subjects[conditioning_subject]
        one_hot_labels = np.eye(6)

        one_hot = one_hot_labels[subj]
        one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
        one_hot = torch.FloatTensor(one_hot).to(device="cuda")

        temp = self.templates.get(subject)

        template = temp.reshape((-1))
        template = np.reshape(template, (-1, template.shape[0]))
        template = torch.FloatTensor(template).to(device="cuda")

        wav_path = audio
        speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft", cache_dir="pretrained_weights")
        audio_feature = processor(speech_array, return_tensors="pt", padding="longest",
                                  sampling_rate=sampling_rate).input_values
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature).to(device="cuda")

        num_frames = int(audio_feature.shape[0] / sampling_rate * args['output_fps'])
        prediction = self.diffusion.p_sample_loop(
            self.model,
            (1, num_frames, args['vertice_dim']),
            clip_denoised=False,
            model_kwargs={
                "cond_embed": audio_feature,
                "one_hot": one_hot,
                "template": template,
            },
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            device="cuda"
        )
        prediction = prediction.squeeze()
        prediction = prediction.detach().cpu().numpy()

        output_path = render_animation(prediction, wav_path)

        return Path(output_path)

