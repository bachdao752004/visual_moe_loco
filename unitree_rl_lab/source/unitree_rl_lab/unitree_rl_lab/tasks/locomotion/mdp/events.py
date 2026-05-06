"""Locomotion-specific event callbacks (domain randomization helpers).

**PhysX view on Articulation**

Despite the name, ``Articulation.root_physx_view`` is the PhysX view for the **entire** articulation
(all rigid links), not only the kinematic root link. That naming is easy to misread, but it is the
correct API for per-body COM and material queries on articulated robots in Isaac Lab.

On ``Articulation``, prefer ``root_physx_view`` over ``physx_view``: many Isaac Lab versions do not
expose ``physx_view`` on articulations, or it refers to a different object.

**COM tensor layout**

``get_coms()`` returns a tensor of shape ``(num_envs, num_bodies, 7)``. The first three columns are
the CoM position offset (x, y, z); remaining columns are orientation (qx, qy, qz, qw). This module
only reads and writes ``[..., :3]``.

**First run (optional sanity check)**

After resolving ``SceneEntityCfg`` (e.g. ``body_names='base'``), confirm ``body_ids`` indices and
tensor shapes match your asset, then remove the prints::

    # print(asset.num_bodies, body_ids, coms.shape)
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _apply_op(
    tensor: torch.Tensor,
    params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "gaussian"],
) -> torch.Tensor:
    lo, hi = params
    if distribution == "uniform":
        delta = torch.empty_like(tensor).uniform_(lo, hi)
    elif distribution == "gaussian":
        mean, std = (lo + hi) / 2, (hi - lo) / 6
        delta = torch.empty_like(tensor).normal_(mean, std)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    if operation == "add":
        return tensor + delta
    elif operation == "scale":
        return tensor * delta
    elif operation == "abs":
        return delta
    else:
        raise ValueError(f"Unknown operation: {operation}")


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "gaussian"] = "uniform",
):
    """Randomize nominal joint positions and sync ``JointPositionAction`` offset when present."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    joint_ids: slice | torch.Tensor = (
        slice(None)
        if isinstance(asset_cfg.joint_ids, slice)
        else torch.as_tensor(asset_cfg.joint_ids, dtype=torch.long, device=asset.device)
    )

    if pos_distribution_params is not None:
        base = asset.data.default_joint_pos[env_ids]
        if not isinstance(joint_ids, slice):
            base = base[:, joint_ids]

        new_pos = _apply_op(base, pos_distribution_params, operation, distribution)

        if isinstance(joint_ids, slice):
            asset.data.default_joint_pos[env_ids] = new_pos
        else:
            asset.data.default_joint_pos[env_ids[:, None], joint_ids] = new_pos

        try:
            action_term = env.action_manager.get_term("JointPositionAction")
        except KeyError:
            return
        if isinstance(joint_ids, slice):
            action_term._offset[env_ids] = new_pos
        else:
            action_term._offset[env_ids[:, None], joint_ids] = new_pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Add uniform CoM offsets per env for the selected bodies.

    ``com_range`` uses keys ``x``, ``y``, ``z`` (each a ``(lo, hi)`` pair in metres). Offsets are
    sampled per environment and broadcast to all ``body_ids`` for that row of ``coms``.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    body_ids: torch.Tensor = (
        torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
        if isinstance(asset_cfg.body_ids, slice)
        else torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")
    )

    ranges = torch.tensor(
        [com_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z")],
        device="cpu",
    )
    rand_offsets = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids), 1, 3),
        device="cpu",
    )

    coms = asset.root_physx_view.get_coms().clone()
    coms[env_ids[:, None], body_ids, :3] += rand_offsets
    asset.root_physx_view.set_coms(coms, env_ids)
