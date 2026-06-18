from .smpl_wrapper import SMPL
from .hmr2 import HMR2
from .discriminator import Discriminator

from ..utils.download import cache_url
from ..configs import CACHE_DIR_4DHUMANS


def download_models(folder=CACHE_DIR_4DHUMANS):
    """Download and extract checkpoints and files for running inference.

    Skips the download entirely if the extracted files are already present.
    The sentinel file used to verify a complete extraction is:
      <folder>/logs/train/multiruns/hmr2/0/model_config.yaml

    If a partial (corrupted) tarball is detected — i.e. the tarball exists
    but the sentinel does not — it is removed so the next run retries cleanly.
    """
    import os

    SENTINEL = os.path.join(folder, "logs/train/multiruns/hmr2/0/model_config.yaml")
    TARBALL  = os.path.join(folder, "hmr2_data.tar.gz")
    REMOTE   = "https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz"

    os.makedirs(folder, exist_ok=True)

    # Already fully extracted — nothing to do.
    if os.path.exists(SENTINEL):
        print(f"[hmr2] hmr2_data already extracted, using cached files at: {folder}")
        return

    # A tarball without a sentinel means a previous extraction was incomplete.
    # Remove it so the download starts fresh.
    if os.path.exists(TARBALL):
        print("[hmr2] Incomplete extraction detected. Removing corrupted tarball.")
        os.remove(TARBALL)

    # Download from remote.
    print("[hmr2] Downloading file: hmr2_data.tar.gz")
    output = cache_url(REMOTE, TARBALL)
    assert os.path.exists(TARBALL), f"Download failed: {TARBALL} does not exist"

    # Extract the tarball.
    print("[hmr2] Extracting hmr2_data.tar.gz ...")
    ret = os.system(f"tar -xf {TARBALL} -C {folder}")
    if ret != 0 or not os.path.exists(SENTINEL):
        if os.path.exists(TARBALL):
            os.remove(TARBALL)
        raise RuntimeError(
            f"[hmr2] Extraction failed (exit={ret}).\n"
            f"Expected sentinel not found: {SENTINEL}"
        )

    print("[hmr2] hmr2_data extraction complete.")

def check_smpl_exists():
    import os
    candidates = [
        f'{CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl',
        f'data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
    ]
    candidates_exist = [os.path.exists(c) for c in candidates]
    if not any(candidates_exist):
        raise FileNotFoundError(f"SMPL model not found. Please download it from https://smplify.is.tue.mpg.de/ and place it at {candidates[1]}")

    # Code edxpects SMPL model at CACHE_DIR_4DHUMANS/data/smpl/SMPL_NEUTRAL.pkl. Copy there if needed
    if (not candidates_exist[0]) and candidates_exist[1]:
        convert_pkl(candidates[1], candidates[0])

    return True

# Convert SMPL pkl file to be compatible with Python 3
# Script is from https://rebeccabilbro.github.io/convert-py2-pickles-to-py3/
def convert_pkl(old_pkl, new_pkl):
    """
    Convert a Python 2 pickle to Python 3
    """
    import dill
    import pickle

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)

DEFAULT_CHECKPOINT=f'{CACHE_DIR_4DHUMANS}/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt'
def load_hmr2(checkpoint_path=DEFAULT_CHECKPOINT):
    from pathlib import Path
    from ..configs import get_config
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192,256]
        model_cfg.freeze()

    # Ensure SMPL model exists
    check_smpl_exists()

    model = HMR2.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model, model_cfg
