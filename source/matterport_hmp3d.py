import habitat_sim

import random
# %matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import cv2
from PIL import Image

# test_scene = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
# test_scene = "/media/rishi/34D61BBAD61B7AF6/matterport/hm3d/versioned_data/hm3d-0.2/hm3d/train/00006-HkseAnWCgqk/HkseAnWCgqk.basis.glb"
# scene_dataset_path = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
# scene_dataset_path = "/media/rishi/34D61BBAD61B7AF6/matterport/hm3d/versioned_data/hm3d-0.2/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

# test_scene = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
# scene_dataset_path = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

test_scene_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
scene_dataset_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"
#
test_scene_v1 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
# # test_scene_v1 = test_scene_v2.replace("/matterport/hm3d/", "/matterport/hm3d_v1/")
scene_dataset_path_v1 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d.scene_dataset_config.json"
# # scene_dataset_path_v1 = scene_dataset_path_v2.replace("/hm3d/scene_datasets/hm3d/", "/hm3d_v1/versioned_data/hm3d-0.1/hm3d/")
#
test_scene = test_scene_v1
scene_dataset_path = scene_dataset_path_v1

# test_scene = ""

HEIGHT = 1024
WIDTH = 1024
SENSOR_HEIGHT_RANGE = [0.75, 1.5]
TURN_ANGLE = 45
HFOV = 90

CENTER_TARGET_OBJECT = "door"
ADD_TO_CENTER_OBJECT = [0.0, 0.0, 0.0]

region_centers = []
# agent_cfg.action_space = {
#     "move_forward": habitat_sim.agent.ActionSpec(
#         "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
#     ),
#     "turn_left": habitat_sim.agent.ActionSpec(
#         "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
#     ),
#     "turn_right": habitat_sim.agent.ActionSpec(
#         "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
#     ),
# }
def get_v1_replacement(scene_glb, scene_dataset_json):
    scene_glb = scene_glb.replace("/matterport/hm3d/", "/matterport/hm3d_v1/")
    scene_dataset_json = scene_dataset_json.replace("/hm3d/scene_datasets/hm3d/", "/hm3d_v1/versioned_data/hm3d-0.1/hm3d/")

    return scene_glb, scene_dataset_json

# test_scene_v1, scene_dataset_path_v1 = get_v1_replacement(test_scene_v2, scene_dataset_path_v2)
#
# test_scene = test_scene_v1
# scene_dataset_path = scene_dataset_path_v1

def display_sample(sample):
    img = sample["rgb"]
    semantic = sample["semantic"]
    # print(sample.keys())
    # print("semantic shape: ", semantic.shape)

    arr = [img, semantic]
    titles = ["rgba", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()


def get_region_centers(scene, verbose=False, center_target_object="door", move_from_center_object=0.5):
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    object_centers = []
    for region in scene.regions:
        # print region
        # print(region)
        print(
            f"Region id:{region.id}, category:{region.category.name()},"
            f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
        )
        for obj in region.objects:
            # print(
            #     f"Object id:{obj.id}, category:{obj.category.name()},"
            #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            # )
            if obj.category.name() == center_target_object:
                object_centers.append(obj.aabb.center)

            count += 1

    return object_centers


def get_scene_images(sim, sim_cfg, limit_output=(360 // TURN_ANGLE) + 1, max_black_per=0.99):
    cnt = 0
    good_images = []
    while cnt < limit_output:
        # print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

        # action_names = list(
        #     sim_cfg.agents[
        #         0
        #     ].action_space.keys()
        # )
        # print("action_names", action_names)
        # action = random.choice(action_names)
        action = "turn_right"
        print("action", action)
        step = sim.step(action)
        sample = {"rgb": step["color_sensor"], "semantic": step["semantic_sensor"]}
        good_images.append(sample)
        display_sample(sample)
        cnt += 1

        # act = input("Enter the action: (default: turn_right)")
        # if act == 'w':
        #     action = "move_forward"
        """
        while True:
            step = sim.step(action)
            # print("step", step.keys())
            # print("step", step)
            step["color_sensor"] = step["color_sensor"][..., :3].copy()

            sample = {"rgb": step["color_sensor"], "semantic": step["semantic_sensor"]}
            # sum_rgb = np.sum(sample["rgb"][..., :3])
            # print("sum_rgb", sum_rgb)

            display_sample(sample)

            # check if 20% of the image is black
            grey_scale = cv2.cvtColor(sample["rgb"], cv2.COLOR_RGB2GRAY)

            grey_scale = 255 - grey_scale
            plt.imshow(grey_scale)
            plt.show()
            sum_rgb = np.sum(grey_scale[grey_scale == 255])

            grey_scale[grey_scale < 255] = 0

            # invert the image
            # An image that is more than 30% back will have a sum more than 255 * shape[0] * shape[1] * 0.3
            max_black = 255 * sample["rgb"].shape[0] * sample["rgb"].shape[1] * max_black_per
            if sum_rgb > max_black:
                print("black image", sum_rgb, max_black)
                # sim.step("turn_right")
                sim.step("move_backward")
                print("move_backward")
                # sim.step("turn_180")
                cnt = 0
                continue
            else:
                break
        
        good_images.append(sample)
        # display_sample(sample)
        cnt += 1
        """

    for sample in good_images:
        display_sample(sample)

    sim.close()

    return good_images


def extract_image_and_mask(
        scene_path,
        scene_dataset_path,
        width=WIDTH,
        height=HEIGHT,
        recursion_level=0,
        center=[0, 1.0, 0],
        region_center_index=0,
        v1_replacement_func=get_v1_replacement
):
    global region_centers

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path

    backend_cfg.scene_dataset_config_file = scene_dataset_path
    # set initial angle
    backend_cfg.default_agent_id = 0

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.3)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=1.0)  # meter
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_ANGLE)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_ANGLE)
        ),
        "turn_180": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=180)
        )
    }

    # print list of all objects in the scene and their ids
    # for obj in sim.semantic_scene.objects:
    #     print(obj.id, obj.category.name())
    # list of actions
    # print(habitat_sim.ActionSpec)

    sensor_height = random.uniform(SENSOR_HEIGHT_RANGE[0], SENSOR_HEIGHT_RANGE[1])
    # sensor_height = 1.0

    if recursion_level == 0:
        position = center
    else:
        # position = region_centers[region_center_index] + ADD_TO_CENTER_OBJECT
        # add 2 lists
        position = [0, 0, 0]
        position[0] = ADD_TO_CENTER_OBJECT[0] + float(region_centers[region_center_index][0])
        position[1] = ADD_TO_CENTER_OBJECT[1] + float(region_centers[region_center_index][1])
        position[2] = ADD_TO_CENTER_OBJECT[2] + float(region_centers[region_center_index][2])

    position = [0, 1.5, 0]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [height, width],
            "position": position,
            # "position": [-0.18384504 , 4.8140173 , -6.712927],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [height, width],
            "position": position,
            # "position": [-0.18384504 , 4.8140173 , -6.712927],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = sensor_uuid
        sensor_spec.sensor_type = sensor_params["sensor_type"]
        sensor_spec.resolution = sensor_params["resolution"]
        sensor_spec.position = sensor_params["position"]
        sensor_spec.hfov = HFOV

        sensor_specs.append(sensor_spec)

    agent_cfg.sensor_specifications = sensor_specs

    sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(sim_cfg)

    scene = sim.semantic_scene

    if recursion_level == 0:
        # initialize region centers
        region_centers = get_region_centers(scene)
        print("region_centers", region_centers)
        extract_image_and_mask(scene_path, scene_dataset_path, recursion_level=recursion_level + 1, region_center_index=region_center_index)
    elif region_center_index < len(region_centers):
        # get new region centers recursively
        print("position", position, "center", region_centers[region_center_index], "region_center_index", region_center_index)
        get_scene_images(sim, sim_cfg)
        input("Press Enter to continue...")
        extract_image_and_mask(scene_path, scene_dataset_path, recursion_level=recursion_level + 1, region_center_index=region_center_index + 1)


extract_image_and_mask(test_scene, scene_dataset_path)
