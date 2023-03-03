# @title Colab Setup and Imports { display-mode: "form" }
# @markdown (double click to see the code)

import math
import os
import random
import shutil
import sys

import git
import imageio
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import d3_40_colors_rgb

# function to display the topdown map
from PIL import Image
import cv2

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

# %cd /content/habitat-sim

# if "google.colab" in sys.modules:
#     # This tells imageio to use the system FFMPEG that has hardware acceleration.
#     os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

#####################################################
# repo = git.Repo(".", search_parent_directories=True)
dir_path = "/home/rishi/programming/AI/experiments/datasets_extractor/"
# %cd $dir_path
data_path = os.path.join(dir_path, "data")
# @markdown Optionally configure the save path for video output:
output_directory = "output/"  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)

# hmp3d_glb_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
hmp3d_glb_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb"
hmp3d_scene_dataset_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"

org_mp_glb_path = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
org_mp_scene_config = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d.scene_dataset_config.json"

glb_path = hmp3d_glb_path_v2
scene_dataset_path = hmp3d_scene_dataset_path_v2
navmesh_path = glb_path.replace(".glb", ".navmesh")

scene_file_def = glb_path
scene_dataset_json_def = scene_dataset_path

rgb_sensor = True
depth_sensor = True
semantic_sensor = True

turn_angle = 45.0

meters_per_pixel = .15
height = 1

DEF_SETTINGS = {
    "width": 1024,  # Spatial resolution of the observations
    "height": 1024,
    "scene": scene_file_def,  # Scene path
    "scene_dataset": scene_dataset_json_def,  # the scene dataset configuration files
    "default_agent": 0,
    "sensor_height": height,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}

LABEL_LISTS = {
    'floor': ['floor', 'rug'],
    'wall': ['wall'],
    'ceiling': ['ceiling'],
}
SUB_NAME_TO_MAIN_NAME = {
    'floor': 'floor',
    'rug': 'floor',
    'wall': 'wall',
    'ceiling': 'ceiling',
}
LABEL_MASK_OUTPUT_NUMBER = {
    'floor': 1,
    'wall': 2,
    'ceiling': 3,
}

############################################################################################################
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_angle)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_angle)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img, semantic_obs]
    titles = ["rgb", "semantic"]

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()


def print_scene_recur(scene, label_lists=LABEL_LISTS, verbose=False):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    label_name_to_seg_indices = {}
    for label in label_lists:
        label_name_to_seg_indices[label] = []

    for obj in scene.objects:
        obj_name = obj.category.name()

        for label in label_lists:
            obj_flag = False

            for sub_label in label_lists[label]:
                if obj_name == sub_label:
                    obj_flag = True
                    break

            if obj_flag:
                if verbose:
                    print(
                        f"Object id:{obj.id}, category:{obj.category.name()},"
                        f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                    )
                pixel_index = int(str(obj.id).split("_")[-1])
                label_name_to_seg_indices[label].append(pixel_index)

    return label_name_to_seg_indices

def equilidian_distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))

def in_list(list_of_lists, item):
    for list_ in list_of_lists:
        if set(item) == set(list_):
            return True
    return False


def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map. The must be in the format of x and y coordinates, so invert x and y if you get the index directly from an image array
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker=".", markersize=4, alpha=0.8)
    plt.show(block=False)


def init_scene(scene_file, scene_dataset_file=scene_dataset_json_def, sim_settings=DEF_SETTINGS):
    sim_settings["scene"] = scene_file
    sim_settings["scene_dataset"] = scene_dataset_file

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    print(print_scene_recur(sim.semantic_scene))

    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
    agent.set_state(agent_state)

    return sim, agent


def get_topdown_map(
        sim,
        scene_name,
        height=height,
        meters_per_pixel=meters_per_pixel,
        display=True
):
    print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    # get bounding box minumum elevation for automatic height

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
        # This map is a 2D boolean array
        sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

        if display:
            # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
            hablab_topdown_map = maps.get_topdown_map(
                sim.pathfinder, height, meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            hablab_topdown_map = recolor_map[hablab_topdown_map]
            print("Displaying the raw map from get_topdown_view:")
            # display_map(sim_topdown_map)
            print("Displaying the map from the Habitat-Lab maps module:")
            display_map(hablab_topdown_map)


def get_unique_heights(sim):
    # @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
    pathfinder_seed = 1  # @param {type:"integer"}
    sim.pathfinder.seed(pathfinder_seed)

    # get 50 random navigable points and get all the unique heights
    random_nav_points = []
    for i in range(50):
        nav_point = sim.pathfinder.get_random_navigable_point()

        random_nav_points.append(nav_point)
        print("Random navigable point : " + str(nav_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

    # get heights alone from the random nav points
    random_nav_points_heights = [point[1] for point in random_nav_points]
    random_nav_points_heights = np.unique(random_nav_points_heights)
    print("Random nav points unique heights: " + str(random_nav_points_heights))

    # keep only one instance of each integer value while keeping the first number's decimals
    unique_heights = []
    unique_heights_prefix = []
    for height in random_nav_points_heights:
        if int(height * 10) not in unique_heights_prefix:
            unique_heights.append(height)
            unique_heights_prefix.append(int(height * 10))

    print("Filtered unique heights: " + str(unique_heights))

    return unique_heights


def recursively_get_nearby_points(
        point, search_pixel_radius=2, recursion_depth=0, nearby_points=[], nearby_points_str_list=[], initial_point=None
):
    if initial_point is None:
        initial_point = point.copy()

    nearby_points_ = []
    # get all the points within a radius of the point
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue

            if {(point[0] + i), (point[1] + j)} == set(initial_point):
                continue

            nearby_points_str = str(point[0] + i) + "_" + str(point[1] + j)
            if nearby_points_str not in nearby_points_str_list:
                nearby_points_str_list.append(nearby_points_str)
                nearby_points_.append([point[0] + i, point[1] + j])
                nearby_points.append([point[0] + i, point[1] + j])

    # recursively call this function on the nearby points
    if recursion_depth < search_pixel_radius - 1:
        # print("Recursion depth: " + str(recursion_depth), "Nearby points: ", len(nearby_points))
        for point_ in nearby_points_:
            recursively_get_nearby_points(
                point_, search_pixel_radius, recursion_depth + 1, nearby_points, nearby_points_str_list, initial_point
            )

    if recursion_depth == 0:
        return nearby_points


# nearby_points = recursively_get_nearby_points([5, 5], search_pixel_radius=2, recursion_depth=0)
# print(nearby_points)
# print(len(nearby_points))
# sys.exit(0)
def make_border_of_mask(
        mask_arr,
        mask_number=1,
        border_mask_number=1,
        border_size=2,
):
    mask_arr = mask_arr.copy()

    # remove all other masks
    mask_arr[mask_arr != mask_number] = 0
    mask_arr[mask_arr == mask_number] = border_mask_number

    # Perform an erosion operation to shrink the object slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_size, border_size))
    eroded_mask = cv2.erode(mask_arr, kernel)

    # Subtract the eroded image from the original image
    border_mask = cv2.absdiff(mask_arr, eroded_mask)

    return border_mask


def get_all_images(
        sim,
        scene_name,
        meters_per_pixel=meters_per_pixel,
        turn_angle=turn_angle,
        save_folder=output_path,
        display=True
):
    print("NavMesh area = " + str(sim.pathfinder.navigable_area))
    print("Bounds = " + str(sim.pathfinder.get_bounds()))

    random_nav_points_heights = get_unique_heights(sim)

    # get the topdown map for these heights
    height_points_list = {}

    # folders are only reset in the subfolder which represents a single scene.
    # This is to avoid deleting the images from the previous iterations
    if not os.path.exists(os.path.join(save_folder + "images", scene_name)):
        os.makedirs(os.path.join(save_folder + "images", scene_name))
    else:
        shutil.rmtree(os.path.join(save_folder + "images", scene_name))
        os.makedirs(os.path.join(save_folder + "images", scene_name))

    if not os.path.exists(os.path.join(save_folder + "masks", scene_name)):
        os.makedirs(os.path.join(save_folder + "masks", scene_name))
    else:
        shutil.rmtree(os.path.join(save_folder + "masks", scene_name))
        os.makedirs(os.path.join(save_folder + "masks", scene_name))

    for i, height in enumerate(random_nav_points_heights):
        # if i > 0:
        #     continue

        height_points_list[height] = []
        sim_topdown_view = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
        hablab_topdown_map = maps.get_topdown_map(
            sim.pathfinder, height, draw_border=True, meters_per_pixel=meters_per_pixel
        )
        print("Displaying the topdown map for height: " + str(height), sim_topdown_view.shape)
        print("Points", hablab_topdown_map, hablab_topdown_map.shape)
        # display_map(sim_topdown_view, key_points=random_nav_points)

        # find index of all 1's in the map

        island_border_points = np.argwhere(hablab_topdown_map == 2)
        print("island border points", len(island_border_points))
        hablab_topdown_map[hablab_topdown_map == 2] = 1
        print("hablab_topdown_map", np.unique(hablab_topdown_map))

        # plt.imshow(hablab_topdown_map)
        # plt.show()

        # add neighbouring points to the neighbouring points list to bad points
        hablab_topdown_map_b = make_border_of_mask(hablab_topdown_map, mask_number=1, border_mask_number=2, border_size=3).copy()
        print("hablab_topdown_map_b", np.unique(hablab_topdown_map_b))
        # plt.imshow(hablab_topdown_map_b)
        # plt.show()

        bad_points = list(np.argwhere(hablab_topdown_map_b == 2))
        print("Nearby neighbour points", len(bad_points), bad_points[:10])

        points = np.argwhere(hablab_topdown_map == 1)
        print("pre-filtered points", len(points))

        # filter pixels that are too close to each other. Minimum distance is 2 pixels
        good_points = []
        for point in points:
            # Can be added to the good points list if it is not in the bad points list.
            # Surrounding points of this point will be added to the bad points list
            if in_list(bad_points, point):
                continue
            else:
                good_points.append([point[1], point[0]])

            nearby_points = recursively_get_nearby_points(point, search_pixel_radius=5)

            # remove the neighbours that are too close to the point
            for nearby_point in nearby_points:
                bad_points.append(nearby_point)

        points = good_points.copy()

        grid_dimensions = (hablab_topdown_map.shape[0], hablab_topdown_map.shape[1])
        # get real world coordinates of the points
        real_points = [
                maps.from_grid(
                    point[1],
                    point[0],
                    grid_dimensions,
                    sim=sim,
                    pathfinder=sim.pathfinder,
                )
                for point in points
            ]
        # real_points = points.copy()
        print("Filtered points", real_points)

        print(f"Points for height: {height}", points, len(points))
        if display:
            display_map(hablab_topdown_map, key_points=points)

        # add height to the points
        for i, point in enumerate(real_points):
            real_points[i] = [point[0], height, point[1]]
        real_points = np.array(real_points)

        display_path_agent_renders = True  # @param{type:"boolean"}
        if display_path_agent_renders:
            print("Rendering observations at path points: num_points = " + str(len(real_points)))
            # tangent = np.array([real_points[1][0] - real_points[0][0], real_points[1][1] - real_points[0][1]])
            agent_state = habitat_sim.AgentState()
            for ix, point in enumerate(real_points):
                # if ix < 10:
                #     continue
                if ix < len(real_points) - 1:
                    point = np.array(point)
                    print("Point: " + str(point))

                    if sim.pathfinder.is_navigable(point):
                        print("Navigable")
                    else:
                        print("Not navigable")
                        continue

                    agent_state.position = point
                    tangent = real_points[ix + 1] - point
                    #
                    tangent_orientation_matrix = mn.Matrix4.look_at(
                        point, point + tangent, np.array([0.0, 1.0, 0.0])
                    )
                    tangent_orientation_q = mn.Quaternion.from_matrix(
                        tangent_orientation_matrix.rotation()
                    )
                    agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                    agent.set_state(agent_state)

                    sub_folder_1 = scene_name + os.sep
                    sub_folder_2 = f"height{str(height)[:5]}_pt{ix}{os.sep}"
                    if not os.path.exists(os.path.join(save_folder + "images", sub_folder_1, sub_folder_2)):
                        os.makedirs(os.path.join(save_folder + "images", sub_folder_1, sub_folder_2))

                    if not os.path.exists(os.path.join(save_folder + "masks", sub_folder_1, sub_folder_2)):
                        os.makedirs(os.path.join(save_folder + "masks", sub_folder_1, sub_folder_2))

                    num_turns = round(360.0 / turn_angle)
                    for i in range(num_turns):
                        observations = sim.step("turn_right")

                        rgb = observations["color_sensor"]
                        semantic = observations["semantic_sensor"]

                        print("rgb", rgb.shape, "semantic", semantic.shape)
                        # get semantic indices and labels

                        # print(sim.semantic_scene.objects)
                        # print(sim.semantic_scene)
                        # print(sim.semantic_scene.levels)
                        # print(sim.semantic_scene.r)
                        object_ids = print_scene_recur(sim.semantic_scene)
                        mask = np.zeros_like(semantic, dtype=np.uint8)


                        for label in LABEL_MASK_OUTPUT_NUMBER:
                            for seg_index in object_ids[label]:
                                mask[semantic == seg_index] = LABEL_MASK_OUTPUT_NUMBER[label]

                        # clean mask
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

                        file = f"{scene_name}_height{str(height)[:5]}_pt{ix}_angle{i * turn_angle}"

                        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(save_folder + "images", sub_folder_1, sub_folder_2, file + ".jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(save_folder + "masks", sub_folder_1, sub_folder_2, file + ".png"), mask)

                        if display:
                            display_sample(rgb_obs=rgb, semantic_obs=mask)


sim, agent = init_scene(scene_file=scene_file_def)
get_all_images(sim, scene_name=scene_file_def.split(os.sep)[-2], display=False)
