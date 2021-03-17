
# blender2 --background --python ~/Projects/autorigging/autorigging/blender_utils/remove_joints.py
import bpy
import os
import sys
import math
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

def resize_obj():
    # mesh coordinate normalization
    obj = bpy.data.objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location.x=0
    obj.location.y=0
    obj.location.z=0
    maxdim=max(obj.dimensions.x, max(obj.dimensions.y, obj.dimensions.z))
    sf=2/maxdim
    bpy.ops.transform.resize(value=(sf, sf, sf))

def resize_and_rotate_obj():
    # mesh coordinate normalization
    obj = bpy.data.objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location.x=0
    obj.location.y=0
    obj.location.z=0
    maxdim=max(obj.dimensions.x, max(obj.dimensions.y, obj.dimensions.z))
    # maxdim = obj.dimensions.x
    # if maxdim != obj.dimensions.y:
    #     __import__('pdb').set_trace()
    #     print('fucki')
    sf= 1/ obj.dimensions.z
    # sf = 2/obj.dimensions.y  # y is the upward direction when loaded by blender
    bpy.ops.transform.resize(value=(sf, sf, sf))
    bpy.ops.transform.rotate(value=math.pi/2, orient_axis='X')
    bpy.ops.transform.translate(value=(0, 0, 0.5))



def decimate_obj(target_vnum = 8000):
    def cleanAllDecimateModifiers(obj):
        for m in obj.modifiers:
            if(m.type=="DECIMATE"):
                print("Removing modifier ")
                obj.modifiers.remove(modifier=m)
    
    def binarysearch(low, high, target_vnum):
        decimateRatio = (low+high)/2
        objectList=bpy.data.objects
        vertexCount = 0
        for obj in objectList:
            if(obj.type=="MESH"):
                # Decimate Start
                cleanAllDecimateModifiers(obj)
                modifier=obj.modifiers.new('DecimateMod','DECIMATE')
                modifier.ratio=1-decimateRatio
                modifier.use_collapse_triangulate=True
                # Decimate End, count the number of vertices
                dg = bpy.context.evaluated_depsgraph_get() #getting the dependency graph
                vertexCount += len(obj.evaluated_get(dg).to_mesh().vertices)
        print("decimateRatio: "+str(decimateRatio))
        print("vertexCount: "+str(vertexCount))
        if(vertexCount <= target_vnum):
            return True
        elif vertexCount < target_vnum:
            return binarysearch(low, decimateRatio, target_vnum)
        else:
            return binarysearch(decimateRatio, high, target_vnum)

    binarysearch(0, 1.0, target_vnum)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

project_path = "/home/whitealex95/Projects/autorigging"
#directories = sorted([f for f in listdir(data_path) if not f.startswith(".")])
source_data_path = "mixamo/unrigged"
dest_data_path = "mixamo/unrigged_normalized_decimated_4k_up"
source_files = sorted([f for f in listdir(join(project_path, source_data_path)) if f.endswith(".obj")])
print(source_files)

for f in source_files:
    source_path = join(project_path, source_data_path, f)
    dest_path = join(project_path, dest_data_path, f)
    # Delete all objects
    clear_scene()
    # Load OBJ
    bpy.ops.import_scene.obj(filepath=source_path)
    # Delete and Merge operation in blender
    # resize_obj()
    resize_and_rotate_obj()
    # Decimate meshes that are too big
    decimate_obj(target_vnum=4000)
    # Save FBX
    # bpy.ops.export_scene.fbx(filepath=dest_path)
    # Save OBJ
    bpy.ops.export_scene.obj(filepath=dest_path.split(".obj")[0]+".obj")
     
    

