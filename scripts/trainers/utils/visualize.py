import os
import numpy as np
import torch
import trimesh
import pyrender
import cv2
from PIL import Image
from tqdm import tqdm

# Set environment variable to use osmesa for headless rendering
os.environ['PYOPENGL_PLATFORM'] = 'egl'

# def load_smplx_uv_from_obj(obj_path: str):
#     """
#     Loads UV vertex coordinates and UV face indices from the official SMPL-X OBJ file.
#     Returns:
#         vt   : (N, 2) UV coords in [0,1]
#         ft   : (F, 3) index triplets for UVs (aligned with mesh faces)
#     """
#     vt = []
#     ft = []

#     with open(obj_path, "r") as f:
#         for line in f:
#             if line.startswith("vt "):  # texture coord
#                 parts = line.strip().split()
#                 vt.append([float(parts[1]), float(parts[2])])

#             elif line.startswith("f "):  # face line with vt references
#                 # Example line format: f v1/vt1 v2/vt2 v3/vt3
#                 parts = line.strip().split()[1:]
#                 vt_ids = [int(p.split('/')[1]) - 1 for p in parts]
#                 ft.append(vt_ids)

#     return np.array(vt, dtype=np.float32), np.array(ft, dtype=np.int32)


def load_smplx_uv_and_faces(obj_path: str):
    """
    Load SMPL-X geometry faces, UV faces, and UV coords from a template OBJ.

    Returns:
        faces_v:  (F, 3) int32 - vertex indices
        faces_vt: (F, 3) int32 - uv vertex indices
        vt:       (N_vt, 2) float32 - UV coordinates in [0,1]
    """
    vt = []
    faces_v = []
    faces_vt = []

    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("vt "):
                parts = line.strip().split()
                vt.append([float(parts[1]), float(parts[2])])

            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                v_idx = []
                vt_idx = []
                for p in parts:
                    tok = p.split("/")
                    v = int(tok[0]) - 1   # vertex index
                    t = int(tok[1]) - 1   # uv index
                    v_idx.append(v)
                    vt_idx.append(t)
                faces_v.append(v_idx)
                faces_vt.append(vt_idx)

    faces_v = np.array(faces_v, dtype=np.int32)
    faces_vt = np.array(faces_vt, dtype=np.int32)
    vt = np.array(vt, dtype=np.float32)

    return faces_v, faces_vt, vt

def bake_texture_to_vertex_colors(new_uv, texture_path: str):
    """
    Sample the texture at each UV to get per-vertex RGBA colors.

    Inputs:
        new_uv:       (M, 2) float – UV in [0,1]
        texture_path: path to PNG/JPG

    Returns:
        vertex_colors: (M, 4) uint8
    """
    tex_img = Image.open(texture_path).convert("RGBA")
    tex_np  = np.array(tex_img)  # H x W x 4, uint8

    H, W, _ = tex_np.shape

    # UV (0,0) is bottom-left in GL convention, but images usually use top-left.
    # Flip V.
    u = new_uv[:, 0]
    v = 1.0 - new_uv[:, 1]

    x = np.clip((u * (W - 1)).round().astype(np.int32), 0, W - 1)
    y = np.clip((v * (H - 1)).round().astype(np.int32), 0, H - 1)

    colors = tex_np[y, x]  # (M, 4)

    return colors

def build_per_vertex_uvs(faces_v, faces_vt, vt, num_vertices):
    """
    Creates a per-vertex UV list where uv[i] corresponds to vertex i.
    Picks the FIRST UV encountered for that vertex.
    """
    vertex_uv = np.zeros((num_vertices, 2), dtype=np.float32)
    seen = np.zeros(num_vertices, dtype=bool)

    F = faces_v.shape[0]
    for f in range(F):
        for i in range(3):
            v = faces_v[f, i]
            t = faces_vt[f, i]
            if not seen[v]:
                vertex_uv[v] = vt[t]
                seen[v] = True

    return vertex_uv


def bake_per_vertex_colors(num_vertices, faces_v, faces_vt, vt, texture_path):
    """
    Bake texture into per-vertex colors for original SMPL-X mesh size.
    """

    # 1. Build per-vertex UV mapping
    vertex_uv = build_per_vertex_uvs(faces_v, faces_vt, vt, num_vertices)

    # 2. Convert UV to colors
    vertex_colors = bake_texture_to_vertex_colors(vertex_uv, texture_path)

    return vertex_colors

def unroll_mesh_for_uv(vertices, faces_v, faces_vt, vt):
    """
    Duplicate vertices so each face corner gets its own vertex+UV.

    Inputs:
        vertices: (N, 3) float - SMPL-X posed verts
        faces_v:  (F, 3) int - indices into vertices
        faces_vt: (F, 3) int - indices into vt
        vt:       (N_vt, 2) float - UV coords

    Returns:
        new_vertices: (F*3, 3)
        new_uv:       (F*3, 2)
        new_faces:    (F, 3)
    """
    F = faces_v.shape[0]

    new_vertices = np.zeros((F * 3, 3), dtype=np.float32)
    new_uv       = np.zeros((F * 3, 2), dtype=np.float32)
    new_faces    = np.zeros((F, 3), dtype=np.int32)

    idx = 0
    for f in range(F):
        for i in range(3):
            new_vertices[idx] = vertices[faces_v[f, i]]
            new_uv[idx]       = vt[faces_vt[f, i]]
            new_faces[f, i]   = idx
            idx += 1

    return new_vertices, new_uv, new_faces


def unroll_uv(faces_v, faces_vt, vt):
    """
    Duplicate vertices so each face corner gets its own vertex+UV.

    Inputs:
        faces_v:  (F, 3) int - indices into vertices
        faces_vt: (F, 3) int - indices into vt
        vt:       (N_vt, 2) float - UV coords

    Returns:
        new_uv:       (F*3, 2)
        new_faces:    (F, 3)
    """
    F = faces_v.shape[0]

    new_uv       = np.zeros((F * 3, 2), dtype=np.float32)
    new_faces    = np.zeros((F, 3), dtype=np.int32)

    idx = 0
    for f in range(F):
        for i in range(3):
            new_uv[idx]       = vt[faces_vt[f, i]]
            new_faces[f, i]   = idx
            idx += 1

    return new_uv, new_faces


def build_textured_trimesh(vertices, faces_v, faces_vt, vt, texture_path):
    new_v, new_uv, new_f = unroll_mesh_for_uv(vertices, faces_v, faces_vt, vt)

    texture_img = Image.open(texture_path).convert("RGBA")

    visuals = trimesh.visual.texture.TextureVisuals(
        uv=new_uv,
        image=texture_img
    )

    mesh = trimesh.Trimesh(
        vertices=new_v,
        faces=new_f,
        visual=visuals,
        process=False
    )
    return mesh


def visualize_smpl(
    smpl_model,
    pose_data, 
    save_path,
    fps=25,
    model = "smplx",
    mesh_color="gray",
    only_face=False,
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
        # # breakpoint()
        smplh_transl = smplh_transl - smplh_transl[:1]  # Center the translation
        # # smplh_transl /= 100.0
        smplh_betas = torch.zeros(smplh_body_pose.shape[0], 16).cuda().float()  # Assuming 16 betas, adjust if needed
        # # breakpoint() 

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
        # breakpoint()
        with torch.no_grad():
            model_output = smpl_model(
                return_verts=True,
                **torch_pose_data
            )

    
    vertices = model_output.vertices.cpu().numpy()
    faces = smpl_model.faces

    uv_template_obj = "/CT/GestureSynth2/work/Embody3D/embody-3d/assets/smplx/smplx_uv.obj"
    texture_path = "/CT/GestureSynth2/work/Embody3D/embody-3d/assets/smplx/smplx_texture_m_alb.png"

    # breakpoint()
    # uv_coords, _ = load_smplx_uv_from_obj(uv_template_obj)
    # assert uv_coords.shape[0] == vertices.shape[1], "UV count must match vertex count."
    faces_v, faces_vt, vt = load_smplx_uv_and_faces(uv_template_obj)

     
    
    # texture_img = Image.open(texture_path).convert("RGBA")
    # material = trimesh.visual.texture.SimpleMaterial(
    #     image=texture_img,
    #     diffuse=[1.0, 1.0, 1.0]
    # )

    # visuals = trimesh.visual.texture.TextureVisuals(
    #     uv=uv_coords,
    #     # face_uvs=uv_faces,
    #     # image=texture_img,
    #     # material=material
    #     image=texture_img,
    # )
    # breakpoint()

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

    if only_face:
        # focus on face only
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.285],     # Y position (height)
            [0.0, 0.0, 1.0, 0.25],    # Z position (distance from character) - negative to be in front
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
        
        
        # Set vertex colors instead of face colors for smooth rendering
        # vertex_colors = np.ones((len(vertices[frame_idx]), 4)) * color_map[mesh_color] #  # 
        # mesh.visual.vertex_colors = vertex_colors

        
        # ------------------- working version v1 -------------------
        # verts_frame = vertices[frame_idx]  # (N, 3)
        # new_v, new_uv, new_f = unroll_mesh_for_uv(
        #     verts_frame,
        #     faces_v,
        #     faces_vt,
        #     vt
        # )
        # vertex_colors = bake_texture_to_vertex_colors(new_uv, texture_path)
        

        # mesh = trimesh.Trimesh(
        #     vertices=new_v,
        #     faces=new_f,
        #     # process=False
        # )
        # mesh.visual.vertex_colors = vertex_colors

        # ------------------- working version v2 -------------------
        if frame_idx == 0:
            vertex_colors = bake_per_vertex_colors(
                num_vertices=vertices[frame_idx].shape[0],
                faces_v=faces_v,
                faces_vt=faces_vt,
                vt=vt,
                texture_path=texture_path
            )
        mesh = trimesh.Trimesh(
            vertices=vertices[frame_idx],
            faces=faces,
            process=False
        )
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