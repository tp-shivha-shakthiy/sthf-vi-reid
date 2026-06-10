"""Tests for YAML config parsing and dataset path handling."""
import os
import yaml
import pytest


CONFIG_PATHS = [
    "configs/baseline_hitsz.yaml",
    "configs/sthf_fixed_hitsz.yaml",
    "configs/sthf_adaptive_hitsz.yaml",
    "configs/cutoff_ablation_hitsz.yaml",
]


def _load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TestConfigStructure:
    def test_all_configs_have_data_section(self):
        for path in CONFIG_PATHS:
            cfg = _load_config(path)
            assert "data" in cfg, f"{path} missing 'data' section"
            assert "root" in cfg["data"], f"{path} missing data.root"
            assert "seq_len" in cfg["data"], f"{path} missing data.seq_len"
            assert "img_size" in cfg["data"], f"{path} missing data.img_size"

    def test_all_configs_have_model_section(self):
        for path in CONFIG_PATHS:
            cfg = _load_config(path)
            assert "model" in cfg, f"{path} missing 'model' section"
            assert "name" in cfg["model"], f"{path} missing model.name"

    def test_all_configs_have_train_section(self):
        for path in CONFIG_PATHS:
            cfg = _load_config(path)
            assert "train" in cfg, f"{path} missing 'train' section"
            assert "epochs" in cfg["train"], f"{path} missing train.epochs"
            assert "lr" in cfg["train"], f"{path} missing train.lr"

    def test_all_configs_have_loss_section(self):
        for path in CONFIG_PATHS:
            cfg = _load_config(path)
            assert "loss" in cfg, f"{path} missing 'loss' section"
            for key in ("lambda_id", "lambda_tri", "lambda_int_id", "lambda_int_tri"):
                assert key in cfg["loss"], f"{path} missing loss.{key}"

    def test_no_legacy_dataset_section(self):
        for path in CONFIG_PATHS:
            cfg = _load_config(path)
            assert "dataset" not in cfg, (
                f"{path} still has old 'dataset' section; "
                f"use data.root / data.seq_len / data.img_size instead"
            )

    def test_data_root_is_relative_path(self):
        for path in CONFIG_PATHS:
            cfg = _load_config(path)
            root = cfg["data"]["root"]
            assert isinstance(root, str), f"{path} data.root is not a string"
            assert not os.path.isabs(root), (
                f"{path} data.root should be relative, got absolute: {root}"
            )

    def test_img_size_is_pair(self):
        for path in CONFIG_PATHS:
            cfg = _load_config(path)
            img_size = cfg["data"]["img_size"]
            assert len(img_size) == 2, f"{path} data.img_size should be [H, W]"
            assert all(isinstance(v, int) for v in img_size)


class TestDatasetMissingPath:
    def test_dataset_constructor_raises_on_missing_root(self):
        from data.hitsz_vcm import HITSZVCM
        with pytest.raises(FileNotFoundError) as excinfo:
            HITSZVCM(root="/nonexistent/path", seq_len=6, split="train")
        assert "Split directory not found" in str(excinfo.value)
        assert "/nonexistent/path" in str(excinfo.value)

    def test_sanity_check_script_fails_on_missing_root(self):
        import subprocess
        result = subprocess.run(
            ["python", "scripts/sanity_check_dataset.py",
             "--config", "configs/baseline_hitsz.yaml"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "Dataset not found at" in result.stdout
