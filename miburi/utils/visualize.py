import os
import numpy as np
import torch
import trimesh
import pyrender
import cv2
from tqdm import tqdm

# Set environment variable to use osmesa for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

def visualize_smpl(
    smpl_model,
    pose_data, 
    save_path,
    fps=25,
    model = "smplx",
    mesh_color="gray"
):
    """
    Visualizes SMPLH pose data using pyrender and saves the rendered video.
    """
    if model == "smplh":
        smplh_body_pose = torch.from_numpy(pose_data["smplh:body_pose"]).cuda().float().reshape(-1, 21*3)
        smplh_global_orient = torch.from_numpy(pose_data["smplh:global_orient"]).cuda().float()
        smplh_lefthand = torch.from_numpy(pose_data["smplh:left_hand_pose"]).cuda().float().reshape(-1, 15*3)
        smplh_righthand = torch.from_numpy(pose_data["smplh:right_hand_pose"]).cuda().float().reshape(-1, 15*3)
        smplh_transl = torch.from_numpy(pose_data["smplh:translation"]).cuda().float()
        smplh_transl = smplh_transl - smplh_transl[:1]  # Center the translation
        # # smplh_transl /= 100.0
        smplh_betas = torch.zeros(smplh_body_pose.shape[0], 16).cuda().float()  # Assuming 16 betas, adjust if needed

        with torch.no_grad():
            model_output = smpl_model(
                betas=smplh_betas,
                global_orient=smplh_global_orient,
                body_pose=smplh_body_pose,
                left_hand_pose=smplh_lefthand,
                right_hand_pose=smplh_righthand,
                transl=smplh_transl,
                return_verts=True
            )
    else:
        pose_data["transl"] = pose_data["transl"] - pose_data["transl"][:1]  # Center the translation
        torch_pose_data = {k: torch.from_numpy(v).cuda().float() for k, v in pose_data.items()}
        with torch.no_grad():
            model_output = smpl_model(
                return_verts=True,
                **torch_pose_data
            )


    vertices = model_output.vertices.cpu().numpy()
    faces = smpl_model.faces

    output_file = save_path
    # render every frame using pyrender
    # Set up the scene and renderer
    scene = pyrender.Scene()

    
    color_map = {
        "gray": [200/255, 200/255, 250/255, 1.0],
        "light_blue": [80/255, 150/255, 250/255, 1.0],
        "light_pink": [250/255, 80/255, 150/255, 1.0],
        "dark_blue": [50/255, 50/255, 200/255, 1.0],
        "dark_pink": [200/255, 50/255, 50/255, 1.0],
        "dark_green": [50/255, 200/255, 50/255, 1.0],
    }
    # [200/255, 200/255, 250/255, 1.0] # [200/255, 80/255, 250/255, 1.0] #[80/255, 150/255, 250/255, 1.0]
    
    # Create camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    # camera_pose = np.array([
    #     [1.0, 0.0, 0.0, 0.0],
    #     [0.0, -1.0, 0.0, -0.0],     # Y position (height)
    #     [0.0, 0.0, -1.0, -3.0],    # Z position (distance from character) - negative to be in front
    #     [0.0, 0.0, 0.0, 1.0]
    # ])
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.2],     # Y position (height)
        [0.0, 0.0, 1.0, 2.0],    # Z position (distance from character) - negative to be in front
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Add lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)
    
    # Create renderer with multiple fallback options
    renderer = None
    platforms = ['egl', 'osmesa', 'glx']
    
    for platform in platforms:
        try:
            os.environ['PYOPENGL_PLATFORM'] = platform
            renderer = pyrender.OffscreenRenderer(640, 480)
            print(f"Successfully created renderer with platform: {platform}")
            break
        except Exception as e:
            print(f"Failed to create renderer with platform {platform}: {e}")
            continue
    
    if renderer is None:
        raise RuntimeError("Failed to create renderer with any platform")
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))
    
    # Render each frame
    mesh_node = None
    for frame_idx in tqdm(range(vertices.shape[0]), desc="Rendering frames"):
        # Create mesh for current frame
        mesh = trimesh.Trimesh(vertices[frame_idx], faces)
        
        # Set vertex colors instead of face colors for smooth rendering
        vertex_colors = np.ones((len(vertices[frame_idx]), 4)) * color_map[mesh_color] #  # 
        mesh.visual.vertex_colors = vertex_colors
        
        # Convert to pyrender mesh (smooth=True by default)
        py_mesh = pyrender.Mesh.from_trimesh(mesh)
        
        # Remove previous mesh if exists
        if mesh_node is not None:
            scene.remove_node(mesh_node)
        
        # Add new mesh to scene
        mesh_node = scene.add(py_mesh)
        
        # Render the scene
        color, depth = renderer.render(scene)
        
        # Convert to BGR for OpenCV
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        
        # Write frame to video
        out.write(color_bgr)
        
        # print(f"Rendered frame {frame_idx + 1}/{vertices.shape[0]}")
    
    # Clean up
    out.release()
    renderer.delete()

    # pass video through ffmpeg to ensure compatibility
    os.system(f"ffmpeg -y -loglevel error -i {output_file}  {output_file.replace('.mp4', '_final.mp4')} ")
    os.remove(output_file)
    os.rename(
        output_file.replace('.mp4', '_final.mp4'), 
        output_file
    )
    

def join_2_videos_horizontally(video1_path, video2_path, output_path):
    """
    Joins two videos horizontally and saves the result.
    """
    cmd = f"ffmpeg -y -loglevel error -i {video1_path} -i {video2_path} -filter_complex hstack {output_path}"
    os.system(cmd)
    # print(f"Joined videos saved to {output_path}")



def add_audio_to_video(video_path, audio_path, output_path):
    """
    Adds audio to a video file and saves the result.
    """
    cmd = f"ffmpeg -y -loglevel error -i {video_path} -i {audio_path} -c:v copy -c:a mp3 {output_path}"
    os.system(cmd)
    # print(f"Video with audio saved to {output_path}")