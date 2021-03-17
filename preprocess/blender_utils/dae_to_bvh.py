import bpy
import os
import math
from os import listdir, mkdir
from os.path import join, exists

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
# get keyframes of object list
# def get_keyframes(obj_list):
#     keyframes = []
#     for obj in obj_list:
#         anim = obj.animation_data
#         if anim is not None and anim.action is not None:
#             for fcu in anim.action.fcurves:
#                 for keyframe in fcu.keyframe_points:
#                     x, y = keyframe.co
#                     print(x,y)
#                     if x not in keyframes:
#                         keyframes.append((math.ceil(x)))
#     return keyframes



project_path = '/home/whitealex95/Projects/autorigging'
# rigged_path = join(project_path, 'mixamo/rerigged_4k')
animation_path = join(project_path, 'mixamo/animated')
# animation_frame_path = join(project_path, 'mixamo/objs_4k')

## Load all animations
animation_models_list = sorted(listdir(animation_path))
# animation_models_list = ['aj']  # select single animation_model
# animation_models_list = ['zombie']
action = ["Back Squat", "Drunk Walk Backwards", "Samba Dancing", "Shoved Reaction With Spin_1", "Shoved Reaction With Spin_2", "Warming Up_1", "Warming Up_2"]
fCount = [18, 13, 274, 45, 45, 95, 95]


for animation_model in animation_models_list:
    # consider only collada animations
    # sorted with lower case first
    animations_list = sorted([ anim for anim in listdir(join(animation_path, animation_model)) if anim.endswith('.dae')])
    print(f"Animations_list of {animation_model}: ", animations_list)

    # if not os.path.exists(join(animation_frame_path, animation_model)):
    #     os.mkdir(join(animation_frame_path, animation_model))
    frame_counts_list= []
    for animation in animations_list:
        # import sys
        # save_stdout = sys.stdout
        # sys.stdout = open('trash', 'w')
        try:
            clear_scene()
            source_path = join(animation_path, animation_model, animation)
            dest_path = join(animation_path, animation_model, animation.split('.dae')[0]+'.bvh')
            # Some additional operations will be done to "bones" when importing collada(.dae) files
            # Documentation says collada only takes care of joints? g
            collada_import_args = {
                'import_units': True,
                'fix_orientation': True,
                'find_chains': True,
                'auto_connect': True,
                'keep_bind_info': True
            }
            bpy.ops.wm.collada_import(filepath=source_path, **collada_import_args)
            n_frames = fCount[action.index(animation.split('.dae')[0])]
            bpy.context.scene.frame_start = 0
            bpy.context.scene.frame_end = n_frames - 1  # start from 0 to n_frames
            bpy.context.scene.render.fps = 30
            bpy.ops.export_anim.bvh(filepath=dest_path, check_existing=False,
                                    frame_start=0, frame_end=n_frames-1, root_transform_only=True)  # fix bone length
        except:
            __import__('pdb').set_trace()
        # print('done something?')
        # sys.stdout = save_stdout
