
# blender2 --background --python ~/Projects/autorigging/autorigging/blender_utils/remove_joints.py
import bpy
import os
import sys
import numpy as np
from tqdm import tqdm

from os import listdir, makedirs, system
from os.path import exists, join

def delete_rig_and_join_mesh():
    armature_name = 0
    mesh_names = []
    for idx, obj in enumerate(bpy.data.objects):
        if obj.type == 'ARMATURE':
            armature_name = obj.name
            print(f"{idx} is ARMATURE")
        elif obj.type == 'MESH':
            mesh_names.append(obj.name)
            print(f"{idx} is MESH")
        elif obj.type == 'EMPTY':
            # Delete Empty Data
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.ops.object.delete()
        else:
            print(obj.type)
            
    print(idx, armature_name, mesh_names)
    
    # Merge all mesh (if needed)
    bpy.ops.object.select_all(action='SELECT')
    bpy.context.view_layer.objects.active=bpy.data.objects[mesh_names[0]]
    bpy.ops.object.join()
    
    # Delete Rigged Info (Including Bind Pose Matrices)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[armature_name].select_set(True)
    bpy.ops.object.delete()

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

project_path = "./Projects/autorigging"
#directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
source_data_path = "mixamo/original"
dest_data_path = "mixamo/unrigged"
source_files = sorted([f for f in listdir(join(project_path, source_data_path)) if f.endswith(".fbx") and not f.endswith("_unrigged.fbx")])

for f in tqdm(source_files):
    source_path = join(project_path, source_data_path, f)
    dest_path = join(project_path, dest_data_path, f)

    # Delete all objects
    clear_scene()
    # Load FBX
    bpy.ops.import_scene.fbx(filepath=source_path)
    # Delete and Merge operation in blender
    delete_rig_and_join_mesh()
    # Save FBX
    bpy.ops.export_scene.fbx(filepath=dest_path)
    # Save OBJ
    bpy.ops.export_scene.obj(filepath=dest_path.split(".fbx")[0]+".obj")
     
    

