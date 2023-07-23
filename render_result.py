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
    tmp_dir = 'C:/tmp'
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

    ref_mesh = trimesh.load_mesh('BIWI/templates/BIWI_topology.obj', process=False)
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
    errs = []
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=(188, 158, 33, 255), metallicFactor=0.4,
                                                  roughnessFactor=0.8)

    colors = [
        (68, 65, 249, 255),
        (44, 114, 243, 255),
        (30, 150, 248, 255),
        (79, 199, 249, 255),
        (109, 190, 144, 255),
        (139, 170, 67, 255),
        (144, 117, 87, 255),

    ]
    material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor= (144, 117, 87, 255),
                metallicFactor=0.8,
                roughnessFactor=0.8,
                wireframe=True,
                emissiveFactor= 0.3,
    )

    audio_path = 'BIWI/wav'
    for directory, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".npy") and "_condition_" in filename:

                subj = filename[:2]
                cond = filename.split("_condition_")[1][:2]

                # if subj != cond:
                #     continue
                # Extract the base filename (without the extension)
                basename = os.path.splitext(filename)[0]
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

def render_single_animation_biwi(tmp_dir, filename, audio_filename):
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
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)

    r = pyrender.OffscreenRenderer(640, 480)
    colors = [
        (68, 65, 249, 255),
        (44, 114, 243, 255),
        (30, 150, 248, 255),
        (79, 199, 249, 255),
        (109, 190, 144, 255),
        (139, 170, 67, 255),
        (144, 117, 87, 255),

    ]
    material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor= (144, 117, 87, 255),
                metallicFactor=0.8,
                roughnessFactor=0.8,
                wireframe=True,
                emissiveFactor= 0.3,
    )

    print(filename)

    video_woA_path = os.path.join(tmp_dir, "tmp.mp4")
    video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))

    frames = np.load(filename)
    frames = frames.reshape((-1, 70110 // 3, 3))

    ref_mesh = trimesh.load_mesh('BIWI/templates/BIWI_topology.obj', process=False)

    for i, frame in enumerate(frames):
        ref_mesh.vertices = frames[i, :, :]
        py_mesh = pyrender.Mesh.from_trimesh(ref_mesh, material=material)

        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0, 0])
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
    video_filename = os.path.join("C:/Users/Stefan/Documents/renders/tuning/videos", "bjork_biwi_2.mp4")

    video_frames = ffmpeg.input(f"{tmp_dir}/tmp.mp4")
    audio = ffmpeg.input(audio_filename)

    ffmpeg.concat(video_frames, audio, v=1, a=1).output(video_filename).run(overwrite_output=True)
    del ref_mesh
    gc.collect()

def render_single_animation_mf(tmp_dir, filename, audio_filename):
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
                            [0.0, 0.0, 1.0, 2],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)

    r = pyrender.OffscreenRenderer(640, 480)
    colors = [
        (68, 65, 249, 255),
        (44, 114, 243, 255),
        (30, 150, 248, 255),
        (79, 199, 249, 255),
        (109, 190, 144, 255),
        (139, 170, 67, 255),
        (144, 117, 87, 255),

    ]
    material = pyrender.material.MetallicRoughnessMaterial(
                alphaMode='BLEND',
                baseColorFactor= (109, 190, 144, 255),
                metallicFactor=0.8,
                roughnessFactor=0.8,
                wireframe=True,
                emissiveFactor= 0.3,
    )

    print(filename)

    video_woA_path = os.path.join(tmp_dir, "tmp.mp4")
    video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))

    frames = np.load(filename)
    frames = frames.reshape((-1, 6172, 3))

    ref_mesh = trimesh.load_mesh('multiface/002643814.obj', process=False)

    for i, frame in enumerate(frames):
        ref_mesh.vertices = frames[i, :, :]
        py_mesh = pyrender.Mesh.from_trimesh(ref_mesh, material=material)

        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0, 0])
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
    video_filename = os.path.join("C:/Users/Stefan/Documents/renders/tuning/videos", "teaser_mf.mp4")

    video_frames = ffmpeg.input(f"{tmp_dir}/tmp.mp4")
    audio = ffmpeg.input(audio_filename)

    ffmpeg.concat(video_frames, audio, v=1, a=1).output(video_filename).run(overwrite_output=True)
    del ref_mesh
    gc.collect()


def render_animation_voca(directory, tmp_dir):
    """
    Renders an animation from NPY files in a directory using pyrender and ffmpeg.

    Parameters:
        directory (str): The directory containing the WAV and NPY files.
        tmp_dir (str): The directory where temporary image files will be stored.

    Returns:
        None
    """

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 2],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)

    r = pyrender.OffscreenRenderer(640, 480)
    errs = []

    audio_path = 'D:/uni/thesis-face/FaceXHuBERT/multiface/wav/'
    for directory, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".npy"):
                # Extract the base filename (without the extension)
                basename = os.path.splitext(filename)[0]
                print(filename)

                video_woA_path = os.path.join(tmp_dir, "tmp.mp4")
                video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))

                # Load the NPY file for the original frames
                audio_filename = os.path.join(audio_path, filename.split('_condition_')[0] + ".wav")

                # _, subj, seq_name, _, cond = filename.split('.')[0].split('_')

                # if subj != cond:
                #     continue

                frames = np.load(os.path.join(directory, filename))
                frames = frames.reshape((-1, 6172, 3))

                print(frames.shape[0])
                material = pyrender.material.MetallicRoughnessMaterial(
                    alphaMode='BLEND',
                    baseColorFactor=(144, 117, 87, 255),
                    metallicFactor=0.8,
                    roughnessFactor=0.8
                )

                ref_mesh = trimesh.load_mesh('multiface/002643814.obj', process=False)
                for i, frame in enumerate(frames):
                    ref_mesh.vertices = frames[i, :, :]
                    py_mesh = pyrender.Mesh.from_trimesh(ref_mesh, material=material)

                    scene = pyrender.Scene(#bg_color=[25.0 / 255.0, 28.0 / 255.0, 38.0 / 255.0],
                                           ambient_light=[0.02, 0.02, 0.02], bg_color=[0, 0, 0, 0])
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
                video_filename = os.path.join("C:/Users/Stefan/Documents/renders/tuning/videos", filename[:-4] + "_v0.mp4")

                video_frames = ffmpeg.input(f"{tmp_dir}/tmp.mp4")
                audio = ffmpeg.input(audio_filename)
                print(audio_filename)

                ffmpeg.concat(video_frames, audio, v=1, a=1).output(video_filename).run(overwrite_output=True)
                del ref_mesh
                gc.collect()


def render_single_animation_voca(tmp_dir, filename, audio_filename):
    """
    Renders an animation from NPY files in a directory using pyrender and ffmpeg.

    Parameters:
        directory (str): The directory containing the WAV and NPY files.
        tmp_dir (str): The directory where temporary image files will be stored.

    Returns:
        None
    """

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 0.35],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    r = pyrender.OffscreenRenderer(640, 480)
    errs = []
    # Extract the base filename (without the extension)
    basename = os.path.splitext(filename)[0]
    print(filename)

    video_woA_path = os.path.join(tmp_dir, "tmp.mp4")
    video = cv2.VideoWriter(video_woA_path, fourcc, fps, (640, 480))

    # _, subj, seq_name, _, cond = filename.split('.')[0].split('_')

    # if subj != cond:
    #     continue

    frames = np.load(filename)
    frames = frames.reshape((-1, 5023, 3))

    print(frames.shape[0])
    material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor= (109, 190, 144, 255),
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    ref_mesh = trimesh.load_mesh('vocaset/FLAME_sample.ply', process=False)
    for i, frame in enumerate(frames):
        ref_mesh.vertices = frames[i, :, :]
        py_mesh = pyrender.Mesh.from_trimesh(ref_mesh, material=material)

        scene = pyrender.Scene(#bg_color=[25.0 / 255.0, 28.0 / 255.0, 38.0 / 255.0],
                               ambient_light=[0.02, 0.02, 0.02], bg_color=[0, 0, 0, 0])
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
    video_filename = os.path.join("C:/Users/Stefan/Documents/renders/tuning/videos", "teaser_voca.mp4")

    video_frames = ffmpeg.input(f"{tmp_dir}/tmp.mp4")
    audio = ffmpeg.input(audio_filename)
    print(audio_filename)

    ffmpeg.concat(video_frames, audio, v=1, a=1).output(video_filename).run(overwrite_output=True)
    del ref_mesh
    gc.collect()

def get_heatmap_color(value, min_val, max_val):
    """Returns the color tuple (r, g, b) based on the scalar value."""
    normalized_value = (value - min_val) / (max_val - min_val)
    if normalized_value < 0.5:
        r = 0
        g = int(510 * normalized_value)
        b = 255
    else:
        r = int(510 * (normalized_value - 0.5))
        g = 255
        b = int(510 * (1 - normalized_value))
    return (r, g, b)

def red(value):
    if value < 0.5:
        return 0
    return int(510 * (value - 0.5))

def green(value):
    if value < 0.5:
        return int(510 * value)
    return 255

def blue(value):
    if value < 0.5:
        return 255
    return int(510 * (1 - value))

def visualize_temporal_statistics(directory):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--pred_path", type=str, default="RUN/BIWI/CodeTalker_s2/result/npy/")
    parser.add_argument("--gt_path", type=str, default="./BIWI/vertices_npy/")
    parser.add_argument("--region_path", type=str, default="BIWI/regions/")
    parser.add_argument("--templates_path", type=str, default="BIWI/templates.pkl")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 3.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    r = pyrender.OffscreenRenderer(640, 480)

    with open(args.templates_path, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    for directory, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".npy"):
                # Extract the base filename (without the extension)
                print(filename)

                comp = os.path.join('../result/diface/', filename)
                if not os.path.exists(comp):
                    continue



                subject = filename.split('_')[0]
                seq_frames = np.load(os.path.join(directory, filename))
                seq_frames = seq_frames.reshape((-1, 18516 // 3, 3))


                motion_vectors = np.linalg.norm(seq_frames[1:, :, :] - seq_frames[:-1, :, :], axis=2)

                # compute the mean and standard deviation of the motion vectors
                mean_motion = np.mean(motion_vectors, axis=0)

                x_norm = (mean_motion - np.min(mean_motion)) / (np.max(mean_motion) - np.min(mean_motion))
                std_motion = np.std(motion_vectors, axis=0)

                seq_frames_2 = np.load(comp)
                seq_frames_2 = seq_frames_2.reshape((-1, 18516 // 3, 3))

                motion_vectors_2 = np.linalg.norm(seq_frames_2[1:, :, :] - seq_frames_2[:-1, :, :], axis=2)

                # compute the mean and standard deviation of the motion vectors
                mean_motion_2 = np.mean(motion_vectors_2, axis=0)

                x_norm = (mean_motion_2 - np.min(mean_motion_2)) / (np.max(mean_motion_2) - np.min(mean_motion_2))
                std_motion = np.std(motion_vectors_2, axis=0)


                ms = pmlab.MeshSet()
                ms.load_new_mesh('multiface/002643814.obj')
                template_mesh = ms.current_mesh()
                gt_mesh = pmlab.Mesh(templates[subject].reshape((18516 // 3, 3)), template_mesh.face_matrix(), template_mesh.vertex_normal_matrix(), v_quality_array = mean_motion)
                ms.add_mesh(gt_mesh)

                # ms.apply_filter('quality_mapper_applier', minqualityval=mean_motion.min(),
                #                 maxqualityval=mean_motion.max(), tfslist=6, brightness=2)
                ms.apply_filter('colorize_by_vertex_quality')
                ms.save_current_mesh(f'BIWI/tmp/tmp.obj', save_vertex_color=True)


                ms.load_new_mesh('multiface/002643814.obj')
                template_mesh = ms.current_mesh()
                gt_mesh = pmlab.Mesh(templates[subject].reshape((18516 // 3, 3)), template_mesh.face_matrix(), template_mesh.vertex_normal_matrix(), v_quality_array = mean_motion_2)
                ms.add_mesh(gt_mesh)

                ms.apply_filter('quality_mapper_applier', minqualityval=mean_motion_2.min(),
                                maxqualityval=mean_motion_2.max(), tfslist=2, brightness=1)
                ms.save_current_mesh(f'BIWI/tmp/tmp2.obj', save_vertex_color=True)

                ref_mesh = trimesh.load_mesh(f'BIWI/tmp/tmp.obj')
                ref_mesh.visual.vertx_colors = np.zeros(ref_mesh.vertices.shape)
                py_mesh = pyrender.Mesh.from_trimesh(ref_mesh)

                ref_mesh = trimesh.load_mesh(f'BIWI/tmp/tmp2.obj')
                ref_mesh.visual.vertx_colors = np.random.uniform(size=(ref_mesh.vertices.shape))
                py_mesh_2 = pyrender.Mesh.from_trimesh(ref_mesh)

                scene = pyrender.Scene()
                node = pyrender.Node(
                    mesh=py_mesh_2,
                    translation=[0, 0, 0]
                )
                scene.add_node(node)

                # node = pyrender.Node(
                #     mesh=py_mesh_2,
                #     translation=[1, 0, 0]
                # )
                # scene.add_node(node)

                scene.add(cam, pose=camera_pose)
                scene.add(light, pose=camera_pose)
                color, _ = r.render(scene)

                output_frame = os.path.join('multiface/heatmaps', f"{filename[:-4]}.png")
                cv2.imwrite(output_frame, color)

                del ref_mesh
                gc.collect()



def visualize_temporal_statistics_2(directory):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--pred_path", type=str, default="RUN/BIWI/CodeTalker_s2/result/npy/")
    parser.add_argument("--gt_path", type=str, default="./BIWI/vertices_npy/")
    parser.add_argument("--region_path", type=str, default="BIWI/regions/")
    parser.add_argument("--templates_path", type=str, default="BIWI/templates.pkl")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414)
    camera_pose = np.array([[1.0, 0, 0.0, 0.00],
                            [0.0, 1.0, 0.0, 0.00],
                            [0.0, 0.0, 1.0, 3.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)

    r = pyrender.OffscreenRenderer(640, 480)

    cmap = matplotlib.cm.get_cmap('Reds')

    with open(args.templates_path, 'rb') as fin:
        templates = pickle.load(fin, encoding='latin1')

    for directory, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".npy"):
                # Extract the base filename (without the extension)
                print(filename)

                comp = os.path.join('../result/diface/', filename)
                if not os.path.exists(comp):
                    continue



                subject = filename.split('_')[0]
                seq_frames = np.load(os.path.join(directory, filename))
                seq_frames = seq_frames.reshape((-1, 18516 // 3, 3))


                motion_vectors = np.linalg.norm(seq_frames[1:, :, :] - seq_frames[:-1, :, :], axis=2)

                # compute the mean and standard deviation of the motion vectors
                mean_motion = np.mean(motion_vectors, axis=0)

                x_norm = (mean_motion - np.min(mean_motion)) / (np.max(mean_motion) - np.min(mean_motion))
                std_motion = np.std(motion_vectors, axis=0)

                seq_frames_2 = np.load(comp)
                seq_frames_2 = seq_frames_2.reshape((-1, 18516 // 3, 3))

                motion_vectors_2 = np.linalg.norm(seq_frames_2[1:, :, :] - seq_frames_2[:-1, :, :], axis=2)

                # compute the mean and standard deviation of the motion vectors
                mean_motion_2 = np.mean(motion_vectors_2, axis=0)

                x_norm = (mean_motion_2 - np.min(mean_motion_2)) / (np.max(mean_motion_2) - np.min(mean_motion_2))
                std_motion = np.std(motion_vectors_2, axis=0)

                ref_mesh = trimesh.load_mesh('multiface/002643814.obj', process=False)
                ref_mesh.vertices = templates[subject].reshape((18516 // 3, 3))

                mean_motion = x_norm
                colors = np.array([(cmap(x + 0.1)[1], cmap(x + 0.1)[2], cmap(x + 0.1)[0]) for x in mean_motion])
                ref_mesh.visual.vertex_colors = colors

                # print(dir(ref_mesh.visual.vertex_colors))
                py_mesh = pyrender.Mesh.from_trimesh(ref_mesh, smooth=False)

                scene = pyrender.Scene()
                node = pyrender.Node(
                    mesh=py_mesh,
                    translation=[0, 0, 0]
                )
                scene.add_node(node)

                # node = pyrender.Node(
                #     mesh=py_mesh_2,
                #     translation=[1, 0, 0]
                # )
                # scene.add_node(node)

                scene.add(cam, pose=camera_pose)
                scene.add(light, pose=camera_pose)
                color, _ = r.render(scene)

                output_frame = os.path.join('multiface/heatmaps', f"{filename[:-4]}.png")
                cv2.imwrite(output_frame, color)

                del ref_mesh
                gc.collect()


if __name__ == '__main__':
    render_animation("result", "renders/tmp")
