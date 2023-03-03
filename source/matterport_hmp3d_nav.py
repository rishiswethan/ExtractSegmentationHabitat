# @title Colab Setup and Imports { display-mode: "form" }
# @markdown (double click to see the code)

import math
import os
import random
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

# repo = git.Repo(".", search_parent_directories=True)
dir_path = "/home/rishi/programming/AI/experiments/datasets_extractor/"
# %cd $dir_path
data_path = os.path.join(dir_path, "data")
# @markdown Optionally configure the save path for video output:
output_directory = "output/"  # @param {type:"string"}
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)

hmp3d_glb_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
hmp3d_scene_dataset_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"

org_mp_glb_path = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
org_mp_scene_config = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d.scene_dataset_config.json"

glb_path = hmp3d_glb_path_v2
scene_dataset_path = hmp3d_scene_dataset_path_v2
navmesh_path = glb_path.replace(".glb", ".navmesh")

display = True  # @param {type:"boolean"}

# import the maps module alone for topdown mapping

# @title Configure Sim Settings
############################################################################################################

# test_scene = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
# mp3d_scene_dataset = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d.scene_dataset_config.json"
test_scene = glb_path
mp3d_scene_dataset = scene_dataset_path

rgb_sensor = True  # @param {type:"boolean"}
depth_sensor = True  # @param {type:"boolean"}
semantic_sensor = True  # @param {type:"boolean"}

turn_angle = 30.0

sim_settings = {
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "scene": test_scene,  # Scene path
    "scene_dataset": mp3d_scene_dataset,  # the scene dataset configuration files
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
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

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()
    # cv2.imshow("Semantic ", np.array(semantic_img))
    # cv2.imshow("RGB ", np.array(rgb_img))
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()


########################################################################################
cfg = make_cfg(sim_settings)
try:
    sim.close()
except NameError:
    pass
sim = habitat_sim.Simulator(cfg)

########################################################################################
def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return


# Print semantic annotation information (id, category, bounding box details)
# about levels, regions and objects in a hierarchical fashion
scene = sim.semantic_scene
print_scene_recur(scene)

############################################################################################################

# the randomness is needed when choosing the actions
random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()
agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

########################################################################################

action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

########################################################################################
# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map. The must be in the format of x and y coordinates, so invert x and y if you get the index directly from an image array
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=5, alpha=0.8)
    plt.show(block=False)


# @markdown ###Configure Example Parameters:
# @markdown Configure the map resolution:
meters_per_pixel = .55  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
# @markdown ---
# @markdown Customize the map slice height (global y coordinate):
custom_height = False  # @param {type:"boolean"}
height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
# @markdown If not using custom height, default to scene lower limit.
# @markdown (Cell output provides scene height range from bounding box for reference.)

print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
if not custom_height:
    # get bounding box minumum elevation for automatic height
    height = sim.pathfinder.get_bounds()[0][1]

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

        # easily save a map to file:
        map_filename = os.path.join(output_path, "top_down_map.png")
        imageio.imsave(map_filename, hablab_topdown_map)

########################################################################################

# @markdown ## Querying the NavMesh

def equilidian_distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))

def in_list(list_of_lists, item):
    for list_ in list_of_lists:
        if set(item) == set(list_):
            return True
    return False

if not sim.pathfinder.is_loaded:
    print("Pathfinder not initialized, aborting.")
else:
    # @markdown NavMesh area and bounding box can be queried via *navigable_area* and *get_bounds* respectively.
    print("NavMesh area = " + str(sim.pathfinder.navigable_area))
    print("Bounds = " + str(sim.pathfinder.get_bounds()))

    # @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
    pathfinder_seed = 1  # @param {type:"integer"}
    sim.pathfinder.seed(pathfinder_seed)
    nav_point = sim.pathfinder.get_random_navigable_point()

    #############
    random_nav_points = []
    for i in range(20):
        nav_point = sim.pathfinder.get_random_navigable_point()

        random_nav_points.append(nav_point)
        print("Random navigable point : " + str(nav_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

    # get heights alone from the random nav points
    random_nav_points_heights = [point[1] for point in random_nav_points]
    random_nav_points_heights = np.unique(random_nav_points_heights)
    print("Random nav points heights: " + str(random_nav_points_heights))

    # get the topdown map for these heights
    height_points_list = {}
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
        points = np.argwhere(hablab_topdown_map == 1)
        print("pre-filtered points", points)
        island_border_points = np.argwhere(hablab_topdown_map == 2)
        print("island border points", island_border_points)

        # filter pixels that are too close to each other. Minimum distance is 2 pixels
        bad_points = []
        border_points = []
        good_points = []
        for point in points:
            # don't check the same point again
            if in_list(bad_points, point):
                continue

            # loop through all 8 neighbours of the point
            nearby_points = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    nearby_points.append([point[0] + i, point[1] + j])

            border_reject_flag = False
            # remove the neighbours that are too close to the point
            for nearby_point in nearby_points:
                bad_points.append(nearby_point)

                # remove the point if it neighbours an island border point
                # if in_list(island_border_points, nearby_point) and (not border_reject_flag):
                #     border_points.append(point)
                #     border_reject_flag = True

            if not border_reject_flag:
                good_points.append([point[1], point[0]])

        # remove bad points from the good points list
        # good_points_ = []
        # for point in good_points:
        #     if not in_list(border_points, point):
        #         good_points_.append([point[1], point[0]])

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
                    # tangent = [real_points[ix + 1][0] - point[0], real_points[ix + 1][1] - point[1]]
                    #
                    # tangent_orientation_matrix = mn.Matrix4.look_at(
                    #     point, point + tangent, np.array([0, 1.0, 0])
                    # )
                    # tangent_orientation_q = mn.Quaternion.from_matrix(
                    #     tangent_orientation_matrix.rotation()
                    # )
                    # agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                    agent.set_state(agent_state)
                    # observations = sim.get_sensor_observations()
                    # print(observations.keys())

                    num_turns = round(360.0 / turn_angle)
                    # num_turns = 4
                    for i in range(num_turns):
                        observations = sim.step("turn_right")

                        rgb = observations["color_sensor"]
                        semantic = observations["semantic_sensor"]
                        depth = observations["depth_sensor"]

                        if display:
                            # display_sample(rgb, semantic, depth)
                            # cv2.imshow("rgb", rgb)
                            # cv2.waitKey(0)
                            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(output_path + "images/" + "rgb_h" + str(height) + "_" + str(ix) + "_" + str(i * turn_angle) + ".jpg", rgb)
