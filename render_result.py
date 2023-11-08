import trimesh
import numpy as np
import cv2
import os
import ffmpeg
import gc
import argparse
import pyrender
import pickle
import pymeshlab as pmlab
import matplotlib


def render_single_sequence(seq_path):
    tmp_dir = 'renders/tmp'
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    r = pyrender.OffscreenRenderer(640, 480)

    basename = os.path.splitext(seq_path)[0]
    print(seq_path)

    video_woA_path = os.path.join("tmp.mp4")
    video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))

    frames = np.load(seq_path)
    frames = frames.reshape((-1, 70110 // 3, 3))

    ref_mesh = trimesh.load_mesh('data/BIWI/templates/BIWI_topology.obj', process=False)
    for i, frame in enumerate(frames):
        ref_mesh.vertices = frames[i, :, :]
        py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)

        scene = pyrender.Scene(bg_color=[1, 1, 1],
                               ambient_light=[0.02, 0.02, 0.02])
        node = pyrender.Node(
            mesh=py_mesh,
            translation=[0, 0, 0]
        )
        scene.add_node(node)

        scene.add(cam, pose=camera_pose)
        scene.add(light, pose=camera_pose)
        color, _ = r.render(scene)

        output_frame = os.path.join(tmp_dir, f"{i:04d}.png")

        cv2.imwrite(output_frame, color)
        frame = cv2.imread(output_frame)
        video.write(color)

    video.release()
    del ref_mesh
    gc.collect()


def render_animation(directory, tmp_dir):
    """
    Renders an animation from NPY files in a directory using pyrender and ffmpeg.

    Parameters:
        directory (str): The directory containing the WAV and NPY files.
        tmp_dir (str): The directory where temporary image files will be stored.

    Returns:
        None
    """

    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.4141)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 1.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)

    r = pyrender.OffscreenRenderer(640, 480)
    material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor= (144, 117, 87, 255),
                metallicFactor=0.8,
                roughnessFactor=0.8,
                wireframe=True,
                emissiveFactor= 0.3,
    )

    audio_path = 'data/BIWI/wav'
    for directory, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".npy") and "_condition_" in filename:

                print(filename)

                video_woA_path = os.path.join(tmp_dir, "tmp.mp4")
                video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))

                # Load the NPY file for the original frames
                audio_filename = os.path.join(audio_path, "_".join(filename.split('_')[0:2]) + ".wav")

                frames = np.load(os.path.join(directory, filename))
                frames = frames.reshape((-1, 70110 // 3, 3))

                ref_mesh = trimesh.load_mesh('data/BIWI/templates/BIWI_topology.obj', process=False)

                for i, frame in enumerate(frames):
                    ref_mesh.vertices = frames[i, :, :]
                    py_mesh = pyrender.Mesh.from_trimesh(ref_mesh, material=material)

                    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0, 255])
                    node = pyrender.Node(
                        mesh=py_mesh,
                        translation=[0, 0, 0]
                    )
                    scene.add_node(node)

                    scene.add(cam, pose=camera_pose)
                    scene.add(light, pose=camera_pose)

                    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)

                    output_frame = os.path.join(tmp_dir, f"{i:04d}.png")

                    cv2.imwrite(output_frame, color)
                    frame = cv2.imread(output_frame)
                    video.write(frame)

                video.release()
                video_filename = os.path.join("renders/videos", filename[:-4] + ".mp4")

                video_frames = ffmpeg.input(f"{tmp_dir}/tmp.mp4")
                audio = ffmpeg.input(audio_filename)

                ffmpeg.concat(video_frames, audio, v=1, a=1).output(video_filename).run(overwrite_output=True)
                del ref_mesh
                gc.collect()


if __name__ == '__main__':
    render_animation("result", "renders/tmp")
