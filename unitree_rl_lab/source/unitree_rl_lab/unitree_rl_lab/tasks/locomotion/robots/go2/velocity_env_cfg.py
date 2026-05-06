import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# --- Actuator / observation latency (sim-to-real style) ------------------------------------------
# Baseline delay is (0, 0). For phase-2 DR, set ``_GO2_ACTUATOR_MAX_DELAY`` to ``_GO2_ACTUATOR_PHASE2_MAX_DELAY``.
_GO2_ACTUATOR_MIN_DELAY = 0
_GO2_ACTUATOR_MAX_DELAY = 0
_GO2_ACTUATOR_PHASE2_MAX_DELAY = 2

# Policy obs: ``history_length`` > 1 stacks past frames per ObsTerm (larger policy input). Not the same as fixed-lag
# obs only; for strict lag use a custom ``ObsTerm``. Default 1 = no stack (Isaac Lab default behavior).
POLICY_OBS_HISTORY_LENGTH = 1

ROBOT_CFG = UNITREE_GO2_CFG.replace(
    actuators={
        "GO2HV": unitree_actuators.UnitreeActuatorCfg_Go2HV(
            joint_names_expr=[".*"],
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
            min_delay=_GO2_ACTUATOR_MIN_DELAY,
            max_delay=_GO2_ACTUATOR_MAX_DELAY,
        ),
    },
)

# Mixed HF + mesh sub-terrains. ``proportion`` values sum to 1.0 (Isaac Lab also normalizes internally).
# Includes: rough slope (HF pyramid surrogate), separate rail vs baffle (two ``MeshRailsTerrainCfg``),
# star / floating ring / repeated thin boxes (bar-like), repeated cylinders & cones/pyramids.
# True sloped+noise combo needs custom mesh.
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.07),
        "hf_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.06,
            noise_range=(0.01, 0.08),
            noise_step=0.01,
            downsampled_scale=0.2,
            border_width=0.25,
        ),
        "hf_wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.05,
            amplitude_range=(0.015, 0.12),
            num_waves=4,
            border_width=0.25,
        ),
        "hf_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),
        # Narrower platform => more sloped area; closest built-in to "sloped + rough" is still smooth HF.
        "hf_rough_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.1, 0.48),
            platform_width=1.2,
            border_width=0.25,
        ),
        "hf_slope_steep": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.35, 0.75),
            platform_width=2.0,
            border_width=0.25,
        ),
        "mesh_stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.05, 0.22),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "mesh_stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.07,
            step_height_range=(0.05, 0.22),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # Low rails ~ "bars" / track edges
        "mesh_rails_low": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.04,
            rail_thickness_range=(0.05, 0.10),
            rail_height_range=(0.04, 0.12),
            platform_width=2.0,
        ),
        # Taller rails ~ "baffles" (must duck under / between)
        "mesh_rails_baffle": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.04,
            rail_thickness_range=(0.08, 0.14),
            rail_height_range=(0.18, 0.32),
            platform_width=2.0,
        ),
        "mesh_star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.03,
            num_bars=5,
            bar_width_range=(0.08, 0.16),
            bar_height_range=(0.06, 0.18),
            platform_width=2.0,
        ),
        "mesh_floating_ring": terrain_gen.MeshFloatingRingTerrainCfg(
            proportion=0.025,
            ring_width_range=(0.12, 0.28),
            ring_height_range=(0.06, 0.16),
            ring_thickness=0.10,
            platform_width=2.0,
        ),
        # Thin repeated boxes ~ parallel stumble bars
        "mesh_bars_repeated": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.04,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=8,
                height=0.05,
                size=(0.42, 0.06),
                max_yx_angle=0.0,
                degrees=True,
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=14,
                height=0.09,
                size=(0.32, 0.05),
                max_yx_angle=10.0,
                degrees=True,
            ),
            platform_width=2.0,
        ),
        "hf_stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.07,
            stone_height_max=0.2,
            stone_width_range=(0.4, 0.9),
            stone_distance_range=(0.05, 0.25),
            holes_depth=-10.0,
            platform_width=2.0,
            border_width=0.25,
        ),
        "mesh_gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0.055,
            gap_width_range=(0.25, 0.85),
            platform_width=3.0,
        ),
        "mesh_pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.05,
            pit_depth_range=(0.25, 0.85),
            platform_width=3.0,
            double_pit=False,
        ),
        "mesh_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.045,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
            holes=False,
        ),
        "hf_discrete": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.035,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.15, 0.45),
            obstacle_height_range=(0.03, 0.22),
            num_obstacles=36,
            platform_width=2.0,
            border_width=0.25,
        ),
        "mesh_repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=0.025,
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=6,
                height=0.07,
                radius=0.12,
                max_yx_angle=5.0,
                degrees=True,
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=12,
                height=0.11,
                radius=0.08,
                max_yx_angle=14.0,
                degrees=True,
            ),
            platform_width=2.0,
        ),
        "mesh_repeated_pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            proportion=0.025,
            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=5,
                height=0.07,
                radius=0.14,
                max_yx_angle=5.0,
                degrees=True,
            ),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=10,
                height=0.10,
                radius=0.09,
                max_yx_angle=12.0,
                degrees=True,
            ),
            platform_width=2.0,
        ),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    lidar_ground = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(-60.0, -60.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=30.0,
        ),
        max_distance=5.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    lidar_forward = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.25)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-60.0, 60.0),
            horizontal_res=7.5,
        ),
        max_distance=5.0,
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Domain randomization: Isaac Lab ``mdp`` + locomotion ``mdp`` (:mod:`unitree_rl_lab.tasks.locomotion.mdp.events`).

    Extra startup terms: base CoM and joint default position (calibration). Observation noise stays on ``ObsTerm``.
    Actuator latency: ``ROBOT_CFG`` → ``min_delay`` / ``max_delay`` on ``GO2HV``. Obs stacking: ``POLICY_OBS_HISTORY_LENGTH``.
    """

    # --- startup ---
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.0, 2.0),
            "dynamic_friction_range": (0.0, 2.0),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    randomize_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.2, 0.2),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
            },
        },
    )

    randomize_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "pos_distribution_params": (-0.1, 0.1),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # --- reset ---
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    # --- interval ---
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Velocity command only (base link). Goal / pose commands are out of scope here."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-1, 1),
            body_height=(-0.10, 0.10),
            swing_height=(0.02, 0.14),
            body_pitch=(-0.30, 0.30),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.4, 0.4),
            ang_vel_z=(-1.0, 1.0),
            body_height=(-0.15, 0.15),
            swing_height=(0.00, 0.18),
            body_pitch=(-0.40, 0.40),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specs with policy (p_t + c_t), critic privileged terms, and LiDAR streams."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations: proprioception p_t + command c_t."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            if POLICY_OBS_HISTORY_LENGTH > 1:
                self.history_length = POLICY_OBS_HISTORY_LENGTH
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations: privileged explicit e_t + implicit contact channel i_t."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        foot_contact_bool = ObsTerm(
            func=mdp.foot_contact_bool, clip=(0.0, 1.0), params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")}
        )
        contact_forces_norm = ObsTerm(
            func=mdp.contact_forces_norm,
            clip=(0.0, 100.0),
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "normalize_by": 100.0},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class LiDARCfg(ObsGroup):
        """LiDAR streams for downstream PointNet/CNN encoder (keep non-concatenated)."""

        ground_rays = ObsTerm(
            func=mdp.lidar_3d_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_ground"), "flatten": False},
        )
        forward_rays = ObsTerm(
            func=mdp.lidar_3d_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_forward"), "flatten": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    lidar: LiDARCfg = LiDARCfg()


@configclass
class RewardsCfg:
    """Reward terms grouped as E1/E2 nominal and E3 safe-loco components."""

    # --- E1/E2: tracking ---
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    termination = RewTerm(func=mdp.termination, weight=-1.0)
    alive = RewTerm(func=mdp.alive, weight=1.0)

    # --- E1/E2: regularization ---
    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.3,
        },
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.002)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-6)
    ang_vel_xy_stability = RewTerm(func=mdp.ang_vel_xy_stability, weight=-0.2)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    front_hip_pos = RewTerm(
        func=mdp.front_hip_pos,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(FR|FL)_hip_.*")},
    )
    rear_hip_pos = RewTerm(
        func=mdp.rear_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(RR|RL)_hip_.*")},
    )
    base_height = RewTerm(func=mdp.base_height, weight=-0.1, params={"target_height": 0.33})
    balance = RewTerm(func=mdp.balance, weight=-2.0e-5)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.01)
    torque_exceed = RewTerm(func=mdp.torque_exceed, weight=-2.0)
    # --- temporarily disabled extra shaping rewards (non-core / non-table) ---
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
    # energy = RewTerm(func=mdp.energy, weight=-2.0e-5)
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
    #     },
    # )
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

    # --- temporarily disabled safe-loco / CBF rewards (non-core / non-table) ---
    # vel_safe = RewTerm(func=mdp.vel_safe, weight=1.0, params={"command_name": "base_velocity"})
    # step_penalty = RewTerm(func=mdp.step_penalty, weight=-0.01)
    # collision_penalty = RewTerm(
    #     func=mdp.collision_penalty,
    #     weight=-20.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 120.0},
    # )
    # cbf_action_penalty = RewTerm(func=mdp.cbf_action_penalty, weight=0.0)
    # cbf_psi_penalty = RewTerm(func=mdp.cbf_psi_penalty, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    root_height_below_minimum = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.15})


@configclass
class CurriculumCfg:
    """Curriculum for velocity tracking + terrain. Swap ``terrain_levels_vel`` when adding goal-based tasks."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(mdp.ang_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.lidar_ground.update_period = self.decimation * self.sim.dt
        self.scene.lidar_forward.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
