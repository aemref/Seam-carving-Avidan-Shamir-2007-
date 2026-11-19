"""Skeleton check: every module imports and exposes the names fixed by the interface contract."""

import importlib

CONTRACT = {
    "seamcarving": ["__version__"],
    "seamcarving.types": ["EnergyFn", "SeamFn"],
    "seamcarving.filters": ["correlate2d", "gaussian_kernel"],
    "seamcarving.energy": ["METHODS", "compute_energy", "make_energy_fn"],
    "seamcarving.seam": ["cumulative_map", "find_vertical_seam", "seam_cost", "is_valid_seam"],
    "seamcarving.carve": ["ShrinkResult", "remove_vertical_seam", "shrink_width", "shrink_height"],
    "seamcarving.enlarge": [
        "insert_vertical_seams",
        "inserted_mask",
        "enlarge_width",
        "enlarge_height",
        "amplify_content",
    ],
    "seamcarving.object_removal": [
        "RemovalResult",
        "apply_mask_weights",
        "seams_needed",
        "remove_object",
    ],
    "seamcarving.baselines": ["scale", "center_crop", "best_crop", "remove_columns"],
    "seamcarving.viz": ["energy_to_image", "draw_seams", "comparison_grid", "save_gif"],
    "seamcarving.resize": ["resize"],
    "seamcarving.cli": ["main"],
    "seamcarving.__main__": [],
    "experiments": [],
    "experiments.common": [
        "IMAGES_DIR",
        "MASKS_DIR",
        "FIG_DIR",
        "TABLE_DIR",
        "list_images",
        "load_image",
        "load_mask",
        "save_figure",
        "save_table",
    ],
}


def test_modules_import_and_expose_contract():
    for module_name, names in CONTRACT.items():
        module = importlib.import_module(module_name)
        for name in names:
            assert hasattr(module, name), f"{module_name}.{name} is missing"
