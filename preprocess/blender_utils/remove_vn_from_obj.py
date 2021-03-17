import bpy
import os
import math
from os import listdir, mkdir
from os.path import join, exists

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
project_path = '/home/whitealex95/Projects/autorigging'
original_obj_path = join(project_path, 'mixamo/unrigged_normalized_decimated_4k_up')
processed_obj_path = join(project_path, 'mixamo/unrigged_normalized_decimated_4k_up_processed')

## Load all animations
obj_list = sorted([ obj for obj in listdir(original_obj_path) if obj.endswith('.obj')])

for obj in obj_list:
    clear_scene()
    source_path = join(original_obj_path, obj)
    dest_path = join(processed_obj_path, obj)
    bpy.ops.import_scene.obj(filepath=source_path)
    obj_export_args = {
        'use_animation': False,
        'use_materials': False, # Do not create .mtl files
        'keep_vertex_order': True,
        'use_triangles': True,
        # Default Settings starting from  below
        'use_blen_objects': True, # Objects as OBJ Objects
        'use_normals': False, # Apply Normals
        'use_uvs': False, # Include UVs
    }
    bpy.ops.export_scene.obj(filepath=dest_path, **obj_export_args)