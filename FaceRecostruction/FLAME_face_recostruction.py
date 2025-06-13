import os
os.environ["PYOPENGL_PLATFORM"] = "egl" 
import numpy as np
import pyrender
import trimesh
import imageio
import shutil
import torch
from FLAME_PyTorch.flame_pytorch.flame import FLAME

# Definizione della classe "config" per configurare i parametri del modello FLAME
class Config:
    def __init__(self):
        self.flame_model_path = '/home/fgastaldi/inferno/assets/FLAME/geometry/generic_model.pkl'
        self.batch_size = 1
        self.use_face_contour = False
        self.use_3D_translation = False
        self.shape_params = 300
        self.expression_params = 100
        self.pose_params = 6
        self.static_landmark_embedding_path = '/home/fgastaldi/inferno/assets/FLAME/geometry/flame_static_embedding.pkl'

config = Config()
flame = FLAME(config) # Viene creato un oggetto FLAME con i parametri di configurazione definiti in "config"

global_pose = torch.tensor(np.load("/andromeda/personal/fgastaldi/facemorphic_output/028c231cd32a48e19613fe653680c11e/AU_51/frames_2/frame_76_11:41:1800/globalpose.npy")).unsqueeze(0).float()
jaw_pose = torch.tensor(np.load("/andromeda/personal/fgastaldi/facemorphic_output/028c231cd32a48e19613fe653680c11e/AU_51/frames_2/frame_76_11:41:1800/jawpose.npy")).unsqueeze(0).float()
pose_params = torch.cat([global_pose, jaw_pose], dim = 1)
expression_params = torch.tensor(np.load("/andromeda/personal/fgastaldi/facemorphic_output/028c231cd32a48e19613fe653680c11e/AU_51/frames_2/frame_76_11:41:1800/exp.npy")).unsqueeze(0).float()
shape_params = torch.tensor(np.load("/andromeda/personal/fgastaldi/facemorphic_output/028c231cd32a48e19613fe653680c11e/AU_51/frames_2/frame_76_11:41:1800/shape.npy")).unsqueeze(0).float()

# Nelle righe di codice sottostanti vengono passati i parametri caricati al modello FLAME
output = flame(
    shape_params = shape_params,
    expression_params = expression_params,
    pose_params = pose_params

)

# Nelle righe di codice sottostanti vengono estratti i vertici e le facce della mesh per poi essere creata
vertices = output[0].detach().cpu().numpy()[0]
faces = flame.faces.astype(np.int32)
mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
#mesh.export('reconstructed_face.obj') # La mesh viene salvata in formato .obj
#print('Mesh salvata correttamente')

# TUTTE LE RIGHE DI CODICE SOTTOSTANTI SERVONO SEMPLICEMENTE PER CREARE LA SCENA DI PYRENDER
# LA MESH VIENE RENDERIZZATA E SALVATA IN UN FILE .PNG

scene = pyrender.Scene()
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_pyrender) 

# Le due righe di codice sottostanti vengono utilizzate per definire l'illuminazione della scena
light = pyrender.DirectionalLight(color = np.ones(3), intensity = 1.0)
scene.add(light, pose=[[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 2],
                       [0, 0, 0, 1]])
scene.ambient_light = np.array([0.1, 0.1, 0.1])

# Le due righe di codice sottostanti servono a posizionare la fotocamera necessaria a fare "una foto" del modello 3D affinché si ottenga un'immagine 2D
camera = pyrender.OrthographicCamera(xmag = 0.15, ymag = 0.15)
scene.add(camera, pose=[[1, 0, 0, 0],  
                        [0, 1, 0, 0],
                        [0, 0, 1, 2],
                        [0, 0, 0, 1]])

# Le righe di codice sottostanti permettono di renderizzare la scena e salvare l'immagine in un file .png
r = pyrender.OffscreenRenderer(512, 512)
color, _ = r.render(scene)
imageio.imwrite("/home/fgastaldi/output_flame_geometry.png", color)

