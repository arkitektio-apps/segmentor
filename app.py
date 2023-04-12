from stardist.models import StarDist3D
from csbdeep.utils import Path, normalize
import sys
import numpy as np
import xarray as xr
from concurrent.futures import ThreadPoolExecutor
import asyncio
from arkitekt import Arkitekt
from fakts.grants.remote.device_code import DeviceCodeGrant
from fakts.grants.remote.base import StaticDiscovery
from fakts import Fakts
from mikro.api.schema import (
    ModelFragment,
    from_xarray,
    RepresentationFragment,
    ContextFragment,
    get_image_image_links,
    LinkableModels,
    create_model,
    RepresentationVariety,
    ModelKind,
)
from rekuest.actors.functional import (
    CompletlyThreadedActor,
)
from pydantic import Field
from arkitekt.tqdm import tqdm as atdqm
from arkitekt import easy
from stardist import (
    fill_label_holes,
    random_label_cmap,
    calculate_extents,
    gputools_available,
)
from stardist import Rays_GoldenSpiral
from stardist.models import StarDist2D
from stardist.matching import matching, matching_dataset
from stardist.models import Config3D, StarDist3D, StarDistData3D
from tqdm import tqdm
import shutil
import uuid
from arkitekt import register
from enum import Enum


class PreTrainedModels(str, Enum):
    STARDIST_ORGANOID_3D = "stardist3"


def random_fliprot(img, mask, axis=None):
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim >= mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1, 2))
    x = random_intensity_change(x)
    return x, y


@register()
def train_stardist_model(
    context: ContextFragment,
    epochs: int = 10,
    patches_per_image: int = 1024,
    validation_split: float = 0.1,
) -> ModelFragment:
    """Train Stardist Model

    Trains a care model according on a specific context.

    Args:
        context (ContextFragment): The context

    Returns:
        ModelFragment: The Model
    """

    training_data_id = f"context_data{context.id}"

    x = get_image_image_links(
        LinkableModels.GRUNNLAG_REPRESENTATION,
        LinkableModels.GRUNNLAG_REPRESENTATION,
        "gt",
        context=context,
    )

    for link in x:
        assert link.y.variety == RepresentationVariety.MASK, "Images need to be a mask"

    axis_norm = (0, 1, 2)  # normalize channels independently

    X = [t.x.data.sel(c=0, t=0).transpose(*"zxy").data.compute() for t in x]
    Y = [t.y.data.sel(c=0, t=0).transpose(*"zxy").data.compute() for t in x]

    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]

    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"

    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(0.15 * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    print("number of images: %3d" % len(X))
    print("- training:       %3d" % len(X_trn))
    print("- validation:     %3d" % len(X_val))

    extents = calculate_extents(Y)
    anisotropy = tuple(np.max(extents) / extents)
    print("empirical anisotropy of labeled objects = %s" % str(anisotropy))
    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 96

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    use_gpu = False and gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D(
        rays=rays,
        grid=grid,
        anisotropy=anisotropy,
        use_gpu=use_gpu,
        n_channel_in=n_channel,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size=(48, 96, 96),
        train_batch_size=2,
    )
    print(conf)
    vars(conf)

    if use_gpu:
        from csbdeep.utils.tf import limit_gpu_memory

        # adjust as necessary: limit GPU memory to be used by TensorFlow to leave some to OpenCL-based computations
        limit_gpu_memory(0.8)
        # alternatively, try this:
        # limit_gpu_memory(None, allow_growth=True)

    model = StarDist3D(conf, name="active_model", basedir="models")

    median_size = calculate_extents(Y, np.median)
    fov = np.array(model._axes_tile_overlap("ZYX"))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print(
            "WARNING: median object size larger than field of view of the neural network."
        )

    for i in tqdm(range(epochs)):
        model.train(
            X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=1
        )

    archive = shutil.make_archive("active_model", "zip", "models/active_model")
    model = create_model(
        "active_model.zip",
        kind=ModelKind.TENSORFLOW,
        name=f"Care Model of {context.name}",
        contexts=[context],
    )
    return model


@register()
def predict_flou2(rep: RepresentationFragment) -> RepresentationFragment:
    """Segment Flou2

    Segments Cells using the stardist flou2 pretrained model

    Args:
        rep (Representation): The Representation.

    Returns:
        Representation: A Representation

    """
    print(f"Called wtih Rep {rep.data.nbytes}")
    assert rep.data.nbytes < 1000 * 1000 * 30 * 1 * 2, "Image is to big to be loaded"

    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    axis_norm = (0, 1, 2)
    x = rep.data.sel(c=0, t=0, z=0).transpose(*"xy").data.compute()
    x = normalize(x)

    labels, details = model.predict_instances(x)

    array = xr.DataArray(labels, dims=list("xy"))

    nana = from_xarray(
        array,
        name="Segmented " + rep.name,
        origins=[rep],
        tags=["segmented"],
        variety=RepresentationVariety.MASK,
    )
    return nana


@register()
def upload_pretrained(pretrained: PreTrainedModels) -> ModelFragment:
    """Upload pretrained Stardist

    Uploads a pretrained startdist model

    Args:
        pretrained (PreTrainedModels): _description_

    Returns:
        ModelFragment: _description_
    """
    if pretrained == PreTrainedModels.STARDIST_ORGANOID_3D:
        archive = shutil.make_archive("active_model", "zip", "models/stardist3")
        model = create_model(
            "active_model.zip",
            kind=ModelKind.TENSORFLOW,
            name=f"Stardist Organoid (2022)",
            contexts=[],
        )

    return model


@register()
def predict_flou2(rep: RepresentationFragment) -> RepresentationFragment:
    """Segment Flou2

    Segments Cells using the stardist flou2 pretrained model

    Args:
        rep (Representation): The Representation.

    Returns:
        Representation: A Representation

    """
    print(f"Called wtih Rep {rep.data.nbytes}")
    assert rep.data.nbytes < 1000 * 1000 * 30 * 1 * 2, "Image is to big to be loaded"

    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    axis_norm = (0, 1, 2)
    x = rep.data.sel(c=0, t=0, z=0).transpose(*"xy").data.compute()
    x = normalize(x)

    labels, details = model.predict_instances(x)

    array = xr.DataArray(labels, dims=list("xy"))

    nana = from_xarray(
        array,
        name="Segmented " + rep.name,
        origins=[rep],
        tags=["segmented"],
        variety=RepresentationVariety.MASK,
    )
    return nana


smodel = StarDist3D(None, name="stardist3", basedir="models")


@register()
def predict_stardist(
    rep: RepresentationFragment,
) -> RepresentationFragment:
    """Predict Stardist

    Segments Cells using the stardist algorithm

    Args:
        rep (Representation): The Representation.

    Returns:
        Representation: A Representation

    """
    print(f"Called wtih Rep {rep.data.nbytes}")

    # model = StarDist3D(name=random_dir)

    axis_norm = (0, 1, 2)
    x = rep.data.sel(c=0, t=0).transpose(*"zxy").data.compute()
    x = normalize(x, 1, 99.8, axis=axis_norm)

    labels, details = smodel.predict_instances(x)

    array = xr.DataArray(labels, dims=list("zxy"))

    nana = from_xarray(
        array,
        name="Segmented " + rep.name,
        origins=[rep],
        tags=["segmented"],
        variety=RepresentationVariety.MASK,
    )

    return nana
