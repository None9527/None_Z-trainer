# -*- coding: utf-8 -*-
"""
Dataset and DataLoader for Z-Image training.

Standalone implementation - no musubi-tuner dependency.
"""

import os
import glob
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

try:
    import toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        toml = None

logger = logging.getLogger(__name__)


class ZImageLatentDataset(Dataset):
    """
    Dataset for loading pre-cached latents and text embeddings.
    Supports multiple datasets and per-dataset resolution filtering.
    """
    
    def __init__(
        self,
        datasets: List[Dict],
        shuffle: bool = True,
        max_sequence_length: int = 512,
        cache_arch: str = "zi",  # 'zi' (Z-Image) or 'lc' (LongCat)
    ):
        super().__init__()
        
        self.datasets = datasets
        self.shuffle = shuffle
        self.max_sequence_length = max_sequence_length
        self.cache_arch = cache_arch
        
        # Determine suffixes based on architecture
        self.latent_pattern = f"*_{cache_arch}.safetensors"
        self.te_suffix = f"_{cache_arch}_te.safetensors"
        
        self.cache_files = []
        self.resolutions = []
        
        for ds_config in datasets:
            cache_dir = Path(ds_config.get('cache_directory', ds_config.get('cache_dir', '')))
            repeats = ds_config.get('num_repeats', 1)
            resolution_limit = ds_config.get('resolution_limit', None)
            
            logger.info(f"Loading dataset from: {cache_dir} (repeats={repeats}, limit={resolution_limit})")
            
            files, res_list = self._load_dataset(cache_dir, resolution_limit)
            
            # Apply repeats
            if repeats > 1:
                files = files * repeats
                res_list = res_list * repeats
            
            self.cache_files.extend(files)
            self.resolutions.extend(res_list)
            
        if len(self.cache_files) == 0:
            raise ValueError("No valid cache files found in any dataset")
            
        logger.info(f"Total samples: {len(self.cache_files)} (max_seq_len={max_sequence_length})")
    
    def _load_dataset(self, cache_dir: Path, resolution_limit: Optional[int]) -> Tuple[List[Tuple[Path, Path]], List[Tuple[int, int]]]:
        """Load files from a single directory and filter by resolution"""
        files = []
        resolutions = []
        
        # Find all latent files
        latent_files = list(cache_dir.rglob(self.latent_pattern))
        
        for latent_path in latent_files:
            # Parse resolution
            res = self._parse_resolution(latent_path.stem)
            
            # Filter by resolution limit
            if resolution_limit:
                h, w = res
                if max(h, w) > resolution_limit:
                    continue
            
            # Find text encoder cache
            te_path = self._find_te_path(latent_path, cache_dir)
            
            if te_path and te_path.exists():
                files.append((latent_path, te_path))
                resolutions.append(res)
            
        return files, resolutions

    def _parse_resolution(self, name: str) -> Tuple[int, int]:
        """Parse resolution from filename (e.g., image_1024x1024_zi)"""
        parts = name.split('_')
        res = (1024, 1024) # Default
        for part in parts:
            if 'x' in part and part.replace('x', '').isdigit():
                try:
                    w, h = map(int, part.split('x'))
                    res = (h, w) # (H, W)
                    break
                except:
                    pass
        return res

    def _find_te_path(self, latent_path: Path, cache_dir: Path) -> Optional[Path]:
        """Construct text encoder cache path"""
        name = latent_path.stem
        parts = name.rsplit('_', 2)
        if len(parts) >= 3:
            base_name = parts[0]
        else:
            base_name = name.rsplit('_', 1)[0]
        
        return latent_path.parent / f"{base_name}{self.te_suffix}"
    
    def __len__(self) -> int:
        return len(self.cache_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        latent_path, te_path = self.cache_files[idx]
        
        # Load latent
        latent_data = load_file(str(latent_path))
        latent_key = next((k for k in latent_data.keys() if k.startswith('latents_')), None)
        if latent_key is None:
            raise ValueError(f"No latent key found in {latent_path}")
        latents = latent_data[latent_key]
        
        # 确保latent尺寸能被patch_size=2整除（为Transformer准备）
        C, H, W = latents.shape
        patch_size = 2
        
        # 计算需要填充的尺寸
        H_padded = ((H + patch_size - 1) // patch_size) * patch_size
        W_padded = ((W + patch_size - 1) // patch_size) * patch_size
        
        if H != H_padded or W != W_padded:
            # 填充latent到合适的尺寸 (left, right, top, bottom)
            latents = torch.nn.functional.pad(
                latents, 
                (0, W_padded - W, 0, H_padded - H),  # (left, right, top, bottom)
                mode='reflect'
            )
        
        # Load text encoder output
        te_data = load_file(str(te_path))
        vl_embed_key = next((k for k in te_data.keys() if 'vl_embed' in k), None)
        if vl_embed_key is None:
            raise ValueError(f"No vl_embed key found in {te_path}")
        vl_embed = te_data[vl_embed_key]
        
        # 截断/填充到 max_sequence_length
        seq_len = vl_embed.shape[0]
        if seq_len > self.max_sequence_length:
            vl_embed = vl_embed[:self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - seq_len
            vl_embed = torch.nn.functional.pad(vl_embed, (0, 0, 0, pad_len), mode='constant', value=0)
        
        result = {
            'latents': latents,
            'vl_embed': vl_embed,
        }
        
        # Load cached DINOv3 embeddings if available
        dino_path = latent_path.with_suffix(".dino.safetensors")
        if dino_path.exists():
            dino_data = load_file(str(dino_path))
            if "dino_emb" in dino_data:
                result["dino_emb"] = dino_data["dino_emb"]   # (P, D)
            if "dino_cls" in dino_data:
                result["dino_cls"] = dino_data["dino_cls"]   # (1, D)
            if "dino_mask" in dino_data:
                result["dino_mask"] = dino_data["dino_mask"] # (gh, gw)
        
        return result


class ControlNetDataset(Dataset):
    """
    Dataset for ControlNet training with pre-cached latents and control images.
    Cache format: {name}_{WxH}_zi_controlnet.safetensors
    """
    
    def __init__(
        self,
        datasets: List[Dict],
        max_sequence_length: int = 512,
        cache_arch: str = "zi",
    ):
        super().__init__()
        
        self.max_sequence_length = max_sequence_length
        self.cache_arch = cache_arch
        
        self.latent_pattern = f"*_{cache_arch}_controlnet.safetensors"
        self.te_suffix = f"_{cache_arch}_te.safetensors"
        
        self.cache_files = []
        self.resolutions = []
        
        for ds_config in datasets:
            cache_dir = Path(ds_config['cache_directory'])
            repeats = ds_config.get('num_repeats', 1)
            
            logger.info(f"Loading ControlNet dataset from: {cache_dir}")
            
            files, res_list = self._load_dataset(cache_dir)
            
            if repeats > 1:
                files = files * repeats
                res_list = res_list * repeats
            
            self.cache_files.extend(files)
            self.resolutions.extend(res_list)
        
        if len(self.cache_files) == 0:
            raise ValueError("No valid ControlNet cache files found")
            
        logger.info(f"ControlNet dataset: {len(self.cache_files)} samples")
    
    def _load_dataset(self, cache_dir: Path):
        files = []
        resolutions = []
        
        latent_files = list(cache_dir.glob(self.latent_pattern))
        
        for latent_path in latent_files:
            # Parse resolution from filename
            parts = latent_path.stem.split('_')
            res = (1024, 1024)
            for part in parts:
                if 'x' in part and part.replace('x', '').isdigit():
                    try:
                        w, h = map(int, part.split('x'))
                        res = (h, w)
                        break
                    except:
                        pass
            
            # Find text encoder cache
            base_name = parts[0]
            te_path = latent_path.parent / f"{base_name}{self.te_suffix}"
            
            if te_path.exists():
                files.append((latent_path, te_path))
                resolutions.append(res)
        
        return files, resolutions
    
    def __len__(self) -> int:
        return len(self.cache_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        latent_path, te_path = self.cache_files[idx]
        
        # Load combined cache file (latent + control_latents)
        data = load_file(str(latent_path))
        
        latent_key = next((k for k in data.keys() if k.startswith('latents_')), None)
        if latent_key is None:
            raise ValueError(f"No latent key found in {latent_path}")
        latents = data[latent_key]
        
        # Control latents are VAE-encoded (matching official pipeline)
        control_key = next((k for k in data.keys() if k.startswith('control_latents_')), None)
        if control_key is None:
            # Fallback: try legacy raw RGB key for backward compatibility
            control_key = 'control_image'
        control_latents = data.get(control_key, None)
        if control_latents is None:
            raise ValueError(f"No control_latents found in {latent_path}")
        
        # Load text encoder output
        te_data = load_file(str(te_path))
        vl_embed_key = next((k for k in te_data.keys() if 'vl_embed' in k), None)
        if vl_embed_key is None:
            raise ValueError(f"No vl_embed key found in {te_path}")
        vl_embed = te_data[vl_embed_key]
        
        # Truncate/pad sequence
        seq_len = vl_embed.shape[0]
        if seq_len > self.max_sequence_length:
            vl_embed = vl_embed[:self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - seq_len
            vl_embed = torch.nn.functional.pad(vl_embed, (0, 0, 0, pad_len))
        
        return {
            'latents': latents,
            'vl_embed': vl_embed,
            'control_latents': control_latents,
        }


class Img2ImgDataset(Dataset):
    """
    Dataset for Img2Img training with pre-cached source and target latents.
    Cache format: {name}_{WxH}_zi_img2img.safetensors
    """
    
    def __init__(
        self,
        datasets: List[Dict],
        max_sequence_length: int = 512,
        cache_arch: str = "zi",
    ):
        super().__init__()
        
        self.max_sequence_length = max_sequence_length
        self.cache_arch = cache_arch
        
        self.latent_pattern = f"*_{cache_arch}_img2img.safetensors"
        self.te_suffix = f"_{cache_arch}_te.safetensors"
        
        self.cache_files = []
        self.resolutions = []
        
        for ds_config in datasets:
            cache_dir = Path(ds_config['cache_directory'])
            repeats = ds_config.get('num_repeats', 1)
            
            logger.info(f"Loading Img2Img dataset from: {cache_dir}")
            
            files, res_list = self._load_dataset(cache_dir)
            
            if repeats > 1:
                files = files * repeats
                res_list = res_list * repeats
            
            self.cache_files.extend(files)
            self.resolutions.extend(res_list)
        
        if len(self.cache_files) == 0:
            raise ValueError("No valid Img2Img cache files found")
            
        logger.info(f"Img2Img dataset: {len(self.cache_files)} samples")
    
    def _load_dataset(self, cache_dir: Path):
        files = []
        resolutions = []
        
        latent_files = list(cache_dir.glob(self.latent_pattern))
        
        for latent_path in latent_files:
            parts = latent_path.stem.split('_')
            res = (1024, 1024)
            for part in parts:
                if 'x' in part and part.replace('x', '').isdigit():
                    try:
                        w, h = map(int, part.split('x'))
                        res = (h, w)
                        break
                    except:
                        pass
            
            base_name = parts[0]
            te_path = latent_path.parent / f"{base_name}{self.te_suffix}"
            
            if te_path.exists():
                files.append((latent_path, te_path))
                resolutions.append(res)
        
        return files, resolutions
    
    def __len__(self) -> int:
        return len(self.cache_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        latent_path, te_path = self.cache_files[idx]
        
        # Load combined cache file (target + source latents)
        data = load_file(str(latent_path))
        
        target_key = next((k for k in data.keys() if k.startswith('target_latents_')), None)
        source_key = next((k for k in data.keys() if k.startswith('source_latents_')), None)
        
        if target_key is None or source_key is None:
            raise ValueError(f"Missing target or source latents in {latent_path}")
        
        target_latents = data[target_key]
        source_latents = data[source_key]
        
        # Load text encoder output
        te_data = load_file(str(te_path))
        vl_embed_key = next((k for k in te_data.keys() if 'vl_embed' in k), None)
        if vl_embed_key is None:
            raise ValueError(f"No vl_embed key found in {te_path}")
        vl_embed = te_data[vl_embed_key]
        
        seq_len = vl_embed.shape[0]
        if seq_len > self.max_sequence_length:
            vl_embed = vl_embed[:self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - seq_len
            vl_embed = torch.nn.functional.pad(vl_embed, (0, 0, 0, pad_len))
        
        return {
            'latents': target_latents,  # 用于训练的目标
            'source_latents': source_latents,  # 用于 img2img 的源
            'vl_embed': vl_embed,
        }


class OmniDataset(Dataset):
    """
    Dataset for Omni multi-image training.
    Cache format: 
      - {name}_{WxH}_zi.safetensors (target latent)
      - {name}_zi_siglip.safetensors (condition SigLIP features)
      - {name}_zi_te.safetensors (text embedding)
    """
    
    def __init__(
        self,
        datasets: List[Dict],
        max_sequence_length: int = 512,
        cache_arch: str = "zi",
        condition_cache_dir: str = None,
    ):
        super().__init__()
        
        self.max_sequence_length = max_sequence_length
        self.cache_arch = cache_arch
        self.condition_cache_dir = Path(condition_cache_dir) if condition_cache_dir else None
        
        self.latent_pattern = f"*_{cache_arch}.safetensors"
        self.te_suffix = f"_{cache_arch}_te.safetensors"
        self.siglip_suffix = f"_{cache_arch}_siglip.safetensors"
        
        self.cache_files = []
        self.resolutions = []
        
        for ds_config in datasets:
            cache_dir = Path(ds_config['cache_directory'])
            repeats = ds_config.get('num_repeats', 1)
            
            logger.info(f"Loading Omni dataset from: {cache_dir}")
            
            files, res_list = self._load_dataset(cache_dir)
            
            if repeats > 1:
                files = files * repeats
                res_list = res_list * repeats
            
            self.cache_files.extend(files)
            self.resolutions.extend(res_list)
        
        if len(self.cache_files) == 0:
            raise ValueError("No valid Omni cache files found")
            
        logger.info(f"Omni dataset: {len(self.cache_files)} samples")
    
    def _load_dataset(self, cache_dir: Path):
        files = []
        resolutions = []
        
        # 只匹配标准 latent 文件 (排除 _controlnet, _img2img 等)
        latent_files = [f for f in cache_dir.glob(self.latent_pattern) 
                       if not any(x in f.stem for x in ['controlnet', 'img2img', 'siglip'])]
        
        for latent_path in latent_files:
            parts = latent_path.stem.split('_')
            res = (1024, 1024)
            for part in parts:
                if 'x' in part and part.replace('x', '').isdigit():
                    try:
                        w, h = map(int, part.split('x'))
                        res = (h, w)
                        break
                    except:
                        pass
            
            base_name = parts[0]
            te_path = latent_path.parent / f"{base_name}{self.te_suffix}"
            
            # 查找 SigLIP 特征 (可选)
            siglip_dir = self.condition_cache_dir or cache_dir
            siglip_path = siglip_dir / f"{base_name}{self.siglip_suffix}"
            
            if te_path.exists():
                files.append((latent_path, te_path, siglip_path if siglip_path.exists() else None))
                resolutions.append(res)
        
        return files, resolutions
    
    def __len__(self) -> int:
        return len(self.cache_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        latent_path, te_path, siglip_path = self.cache_files[idx]
        
        # Load target latent
        latent_data = load_file(str(latent_path))
        latent_key = next((k for k in latent_data.keys() if k.startswith('latents_')), None)
        if latent_key is None:
            raise ValueError(f"No latent key found in {latent_path}")
        latents = latent_data[latent_key]
        
        # Load text encoder output
        te_data = load_file(str(te_path))
        vl_embed_key = next((k for k in te_data.keys() if 'vl_embed' in k), None)
        if vl_embed_key is None:
            raise ValueError(f"No vl_embed key found in {te_path}")
        vl_embed = te_data[vl_embed_key]
        
        seq_len = vl_embed.shape[0]
        if seq_len > self.max_sequence_length:
            vl_embed = vl_embed[:self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - seq_len
            vl_embed = torch.nn.functional.pad(vl_embed, (0, 0, 0, pad_len))
        
        result = {
            'latents': latents,
            'vl_embed': vl_embed,
        }
        
        # Load SigLIP features if available
        if siglip_path and siglip_path.exists():
            siglip_data = load_file(str(siglip_path))
            siglip_feats = siglip_data.get('siglip_feats', None)
            if siglip_feats is not None:
                result['siglip_feats'] = siglip_feats
        
        return result


class MultiChannelDataset(Dataset):
    """
    Unified multi-channel dataset for ControlNet / Img2Img / Omni training.

    Replaces separate ControlNetDataset, Img2ImgDataset, OmniDataset with
    a single channel-based architecture.

    Channel config format:
        channels = [
            {"name": "target",  "role": "target",    "suffix": "_zi",           "encoder": "vae"},
            {"name": "source",  "role": "source",    "suffix": "_zi_img2img",   "encoder": "vae"},
            {"name": "depth",   "role": "condition", "suffix": "_zi_controlnet","encoder": "raw"},
            {"name": "ref",     "role": "reference", "suffix": "_zi_siglip",    "encoder": "siglip"},
        ]

    Cache file naming:
        {base_name}_{WxH}{suffix}.safetensors  (per channel)
        {base_name}_zi_te.safetensors          (text embedding, shared)
    """

    # Pre-defined channel presets for backward compatibility
    PRESETS = {
        "img2img": [
            {"name": "target",  "role": "target", "key_prefix": "target_latents_", "suffix": "_zi_img2img"},
            {"name": "source",  "role": "source", "key_prefix": "source_latents_", "suffix": "_zi_img2img"},
        ],
        "controlnet": [
            {"name": "target",  "role": "target",    "key_prefix": "latents_",     "suffix": "_zi_controlnet"},
            {"name": "control", "role": "condition",  "key_prefix": "control_latents_", "suffix": "_zi_controlnet"},
        ],
        "inpaint": [
            {"name": "target",  "role": "target",    "key_prefix": "latents_",         "suffix": "_zi_inpaint"},
            {"name": "masked",  "role": "condition",  "key_prefix": "masked_latents_",  "suffix": "_zi_inpaint"},
            {"name": "mask",    "role": "mask",       "key_prefix": "mask_",            "suffix": "_zi_inpaint"},
        ],
        "omni": [
            {"name": "target",     "role": "target",     "key_prefix": "latents_",      "suffix": "_zi_omni"},
            {"name": "conditions", "role": "conditions",  "key_prefix": "cond_",         "suffix": "_zi_omni", "dynamic": True},
            {"name": "siglip",     "role": "reference",   "key_prefix": "siglip_feats",  "suffix": "_zi_siglip"},
        ],
    }

    def __init__(
        self,
        datasets: List[Dict],
        channels: List[Dict],
        max_sequence_length: int = 512,
        cache_arch: str = "zi",
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.cache_arch = cache_arch
        self.channels = channels
        self.te_suffix = f"_{cache_arch}_te.safetensors"

        # Identify target channel (required)
        self.target_channel = None
        self.aux_channels = []
        for ch in channels:
            if ch.get("role") == "target":
                self.target_channel = ch
            else:
                self.aux_channels.append(ch)

        if self.target_channel is None:
            raise ValueError("MultiChannelDataset requires exactly one channel with role='target'")

        # Determine target file pattern
        target_suffix = self.target_channel.get("suffix", f"_{cache_arch}")
        self.target_pattern = f"*{target_suffix}.safetensors"

        self.cache_files = []  # List of dicts: {"target": Path, "te": Path, "ch_name": Path|None, ...}
        self.resolutions = []

        for ds_config in datasets:
            cache_dir = Path(ds_config['cache_directory'])
            repeats = ds_config.get('num_repeats', 1)
            resolution_limit = ds_config.get('resolution_limit', None)

            logger.info(f"Loading MultiChannel dataset from: {cache_dir}")
            logger.info(f"  Channels: {[ch['name'] for ch in channels]}")

            files, res_list = self._load_dataset(cache_dir, resolution_limit)

            if repeats > 1:
                files = files * repeats
                res_list = res_list * repeats

            self.cache_files.extend(files)
            self.resolutions.extend(res_list)

        if len(self.cache_files) == 0:
            raise ValueError("No valid MultiChannel cache files found")

        logger.info(f"MultiChannel dataset: {len(self.cache_files)} samples, "
                    f"{len(channels)} channels")

    def _load_dataset(self, cache_dir: Path, resolution_limit: Optional[int] = None):
        files = []
        resolutions = []

        # Find all target files
        target_suffix = self.target_channel.get("suffix", f"_{self.cache_arch}")
        exclude_tags = ['controlnet', 'img2img', 'inpaint', 'omni', 'siglip']

        # If target suffix is plain "_zi", exclude specialized files
        if target_suffix == f"_{self.cache_arch}":
            target_files = [
                f for f in cache_dir.glob(self.target_pattern)
                if not any(tag in f.stem for tag in exclude_tags)
            ]
        else:
            target_files = list(cache_dir.glob(self.target_pattern))

        for target_path in target_files:
            res = self._parse_resolution(target_path.stem)

            if resolution_limit:
                h, w = res
                if max(h, w) > resolution_limit:
                    continue

            base_name = self._extract_base_name(target_path.stem)
            te_path = target_path.parent / f"{base_name}{self.te_suffix}"

            if not te_path.exists():
                continue

            # Build file entry for all channels
            entry = {"target": target_path, "te": te_path}

            for ch in self.aux_channels:
                ch_suffix = ch.get("suffix", "")
                if ch_suffix:
                    # Channel data is in a separate file
                    ch_path = cache_dir / f"{base_name}{ch_suffix}.safetensors"
                    entry[ch["name"]] = ch_path if ch_path.exists() else None
                else:
                    # Channel data is embedded in the target file
                    entry[ch["name"]] = "embedded"

            files.append(entry)
            resolutions.append(res)

        return files, resolutions

    def _parse_resolution(self, name: str) -> Tuple[int, int]:
        parts = name.split('_')
        for part in parts:
            if 'x' in part and part.replace('x', '').isdigit():
                try:
                    w, h = map(int, part.split('x'))
                    return (h, w)
                except:
                    pass
        return (1024, 1024)

    def _extract_base_name(self, stem: str) -> str:
        parts = stem.split('_')
        # Remove resolution and suffix parts, keep original name
        # e.g. "cat_1024x1024_zi" -> "cat"
        result_parts = []
        for part in parts:
            if 'x' in part and part.replace('x', '').isdigit():
                break
            result_parts.append(part)
        return '_'.join(result_parts) if result_parts else parts[0]

    def __len__(self) -> int:
        return len(self.cache_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.cache_files[idx]

        # Load target latents
        target_data = load_file(str(entry["target"]))
        target_key_prefix = self.target_channel.get("key_prefix", "latents_")
        latent_key = next((k for k in target_data.keys() if k.startswith(target_key_prefix)), None)
        if latent_key is None:
            raise ValueError(f"No key with prefix '{target_key_prefix}' in {entry['target']}")
        latents = target_data[latent_key]

        # Pad latents to be divisible by patch_size=2
        C, H, W = latents.shape
        patch_size = 2
        H_padded = ((H + patch_size - 1) // patch_size) * patch_size
        W_padded = ((W + patch_size - 1) // patch_size) * patch_size
        if H != H_padded or W != W_padded:
            latents = torch.nn.functional.pad(
                latents, (0, W_padded - W, 0, H_padded - H), mode='reflect'
            )

        # Load text embedding
        te_data = load_file(str(entry["te"]))
        vl_embed_key = next((k for k in te_data.keys() if 'vl_embed' in k), None)
        if vl_embed_key is None:
            raise ValueError(f"No vl_embed key in {entry['te']}")
        vl_embed = te_data[vl_embed_key]

        seq_len = vl_embed.shape[0]
        if seq_len > self.max_sequence_length:
            vl_embed = vl_embed[:self.max_sequence_length]
        elif seq_len < self.max_sequence_length:
            pad_len = self.max_sequence_length - seq_len
            vl_embed = torch.nn.functional.pad(vl_embed, (0, 0, 0, pad_len), mode='constant', value=0)

        result = {
            'latents': latents,
            'vl_embed': vl_embed,
        }

        # Load auxiliary channel data
        for ch in self.aux_channels:
            ch_name = ch["name"]
            ch_path = entry.get(ch_name)
            key_prefix = ch.get("key_prefix", ch_name)

            if ch_path is None:
                # Channel file missing for this sample, skip
                continue

            if ch_path == "embedded":
                # Data is in the target file
                ch_data_val = target_data.get(key_prefix)
                if ch_data_val is not None:
                    result[ch_name] = ch_data_val
            else:
                # Data is in a separate file
                try:
                    ch_data = load_file(str(ch_path))
                    # Try exact key first, then prefix match
                    if key_prefix in ch_data:
                        result[ch_name] = ch_data[key_prefix]
                    else:
                        ch_key = next((k for k in ch_data.keys() if k.startswith(key_prefix)), None)
                        if ch_key:
                            result[ch_name] = ch_data[ch_key]
                except Exception as e:
                    logger.warning(f"Failed to load channel '{ch_name}' from {ch_path}: {e}")

        return result

    @classmethod
    def from_preset(cls, preset_name: str, datasets: List[Dict], **kwargs) -> "MultiChannelDataset":
        """Create from a preset (img2img, controlnet, omni)."""
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(cls.PRESETS.keys())}")
        channels = cls.PRESETS[preset_name]
        return cls(datasets=datasets, channels=channels, **kwargs)


class BucketBatchSampler(torch.utils.data.Sampler):
    """
    支持分桶的 Batch Sampler。
    将具有相同分辨率的样本组合在一起。
    """
    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 按分辨率分组索引
        self.buckets = {} # (h, w) -> [indices]
        for idx, res in enumerate(dataset.resolutions):
            if res not in self.buckets:
                self.buckets[res] = []
            self.buckets[res].append(idx)
            
    def __iter__(self):
        batches = []
        for res, indices in self.buckets.items():
            if self.shuffle:
                # 打乱桶内索引
                indices = torch.tensor(indices)[torch.randperm(len(indices))].tolist()
            
            # 生成 batch
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        
        if self.shuffle:
            # 打乱 batch 顺序
            import random
            random.shuffle(batches)
            
        for batch in batches:
            yield batch

    def __len__(self):
        count = 0
        for indices in self.buckets.values():
            if self.drop_last:
                count += len(indices) // self.batch_size
            else:
                count += (len(indices) + self.batch_size - 1) // self.batch_size
        return count


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    自定义 collate 函数。支持不同分辨率的 latent（自动 padding）。
    """
    # 检查是否所有 latents 具有相同形状
    shapes = [item['latents'].shape for item in batch]
    all_same = all(s == shapes[0] for s in shapes)
    
    if all_same:
        # 所有形状相同，直接 stack
        latents = torch.stack([item['latents'] for item in batch])
    else:
        # 形状不同，需要 padding 到最大尺寸
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)
        
        # 确保尺寸能被 patch_size=2 整除
        patch_size = 2
        max_h = ((max_h + patch_size - 1) // patch_size) * patch_size
        max_w = ((max_w + patch_size - 1) // patch_size) * patch_size
        
        padded_latents = []
        for item in batch:
            lat = item['latents']
            c, h, w = lat.shape
            if h < max_h or w < max_w:
                # Pad to max size (right and bottom padding)
                lat = torch.nn.functional.pad(
                    lat,
                    (0, max_w - w, 0, max_h - h),
                    mode='constant',
                    value=0
                )
            padded_latents.append(lat)
        
        latents = torch.stack(padded_latents)
        logger.debug(f"Padded latents from {shapes} to {latents.shape}")
    
    vl_embeds = [item['vl_embed'] for item in batch]  # 保持 list 形式
    
    return {
        'latents': latents,
        'vl_embed': vl_embeds,
    }


def multi_channel_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate function for MultiChannelDataset.
    Stacks latents, keeps vl_embed as list, collects channel data.
    """
    # Stack latents (same resolution within a bucket)
    shapes = [item['latents'].shape for item in batch]
    all_same = all(s == shapes[0] for s in shapes)

    if all_same:
        latents = torch.stack([item['latents'] for item in batch])
    else:
        max_h = max(s[1] for s in shapes)
        max_w = max(s[2] for s in shapes)
        patch_size = 2
        max_h = ((max_h + patch_size - 1) // patch_size) * patch_size
        max_w = ((max_w + patch_size - 1) // patch_size) * patch_size
        padded = []
        for item in batch:
            lat = item['latents']
            c, h, w = lat.shape
            if h < max_h or w < max_w:
                lat = torch.nn.functional.pad(lat, (0, max_w - w, 0, max_h - h), mode='constant', value=0)
            padded.append(lat)
        latents = torch.stack(padded)

    result = {
        'latents': latents,
        'vl_embed': [item['vl_embed'] for item in batch],
    }

    # Collect all auxiliary channel keys present in any batch item
    aux_keys = set()
    for item in batch:
        aux_keys.update(k for k in item.keys() if k not in ('latents', 'vl_embed'))

    for key in aux_keys:
        values = [item.get(key) for item in batch]
        # Try stacking if all present and same shape, otherwise keep as list
        if all(v is not None for v in values):
            try:
                if all(v.shape == values[0].shape for v in values):
                    result[key] = torch.stack(values)
                else:
                    result[key] = values
            except (AttributeError, RuntimeError):
                result[key] = values
        else:
            result[key] = values

    return result


def create_dataloader(args) -> DataLoader:
    """
    从配置创建 DataLoader。

    支持 dataset_type:
        - "standard" (默认): ZImageLatentDataset
        - "img2img": MultiChannelDataset (img2img preset)
        - "controlnet": MultiChannelDataset (controlnet preset)
        - "omni": MultiChannelDataset (omni preset)
        - "multi_channel": MultiChannelDataset (custom channels config)

    Args:
        args: 训练参数，包含 dataset_config 和其他相关配置

    Returns:
        DataLoader: 数据加载器
    """
    # 读取 dataset 配置
    if hasattr(args, 'dataset_config') and args.dataset_config:
        if isinstance(args.dataset_config, dict):
            # Already parsed (from merged TOML config)
            config = args.dataset_config.copy()
            # Normalize: rename 'sources' to 'datasets' if needed
            if 'sources' in config:
                config['datasets'] = config.pop('sources')
        else:
            config = _read_dataset_config(args.dataset_config)
    else:
        config = {}

    # 获取参数
    datasets = config.get('datasets', [])

    # 兼容旧配置
    if not datasets:
        cache_dir = config.get('cache_directory', getattr(args, 'cache_directory', None))
        if cache_dir:
            datasets = [{
                'cache_directory': cache_dir,
                'num_repeats': config.get('num_repeats', getattr(args, 'num_repeats', 1)),
                'resolution_limit': config.get('resolution_limit', None)
            }]

    if not datasets:
        raise ValueError("No datasets configured. Please check dataset_config.toml or arguments.")

    batch_size = config.get('batch_size', getattr(args, 'batch_size', 1))
    shuffle = config.get('shuffle', getattr(args, 'shuffle', True))
    num_workers = config.get('num_workers', getattr(args, 'num_workers', 4))
    max_sequence_length = config.get('max_sequence_length', getattr(args, 'max_sequence_length', 512))
    cache_arch = config.get('cache_arch', getattr(args, 'cache_arch', 'zi'))

    if getattr(args, 'disable_bucket', False):
        enable_bucket = False
    else:
        enable_bucket = config.get('enable_bucket', getattr(args, 'enable_bucket', True))

    # Determine dataset type
    dataset_type = config.get('dataset_type', getattr(args, 'dataset_type', 'standard'))
    logger.info(f"📋 Dataset type: {dataset_type}")

    # Create dataset based on type
    if dataset_type in ('img2img', 'controlnet', 'omni'):
        # Use preset
        dataset = MultiChannelDataset.from_preset(
            preset_name=dataset_type,
            datasets=datasets,
            max_sequence_length=max_sequence_length,
            cache_arch=cache_arch,
        )
        active_collate = multi_channel_collate_fn

    elif dataset_type == 'multi_channel':
        # Custom channels from config
        channels = config.get('channels', [])
        if not channels:
            raise ValueError("dataset_type='multi_channel' requires 'channels' config")
        dataset = MultiChannelDataset(
            datasets=datasets,
            channels=channels,
            max_sequence_length=max_sequence_length,
            cache_arch=cache_arch,
        )
        active_collate = multi_channel_collate_fn

    else:
        # Standard (backward compatible)
        dataset = ZImageLatentDataset(
            datasets=datasets,
            max_sequence_length=max_sequence_length,
            cache_arch=cache_arch,
        )
        active_collate = collate_fn

    # Create DataLoader
    if enable_bucket and hasattr(dataset, 'resolutions'):
        logger.info("🌊 启用分桶 (BucketBatchSampler)")
        batch_sampler = BucketBatchSampler(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=shuffle
        )
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=active_collate,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=active_collate,
            pin_memory=True,
            drop_last=True,
        )

    logger.info("📦 DataLoader 创建完成")
    return dataloader


def _read_dataset_config(config_path: str) -> dict:
    """
    读取 dataset 配置文件，支持多种格式：
    
    1. 合并格式 (新): [dataset] + [[dataset.sources]] 在主配置中
    2. 独立格式 (旧): [general] + [[datasets]] 在单独文件中
    3. 旧格式: [dataset] 块
    """
    if toml is None:
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    # 1. 合并格式 (新): [dataset] + [[dataset.sources]] 
    #    主配置文件中的 dataset 块
    if 'dataset' in config:
        dataset_config = config['dataset'].copy()
        # 将 sources 重命名为 datasets (兼容 create_dataloader)
        if 'sources' in dataset_config:
            dataset_config['datasets'] = dataset_config.pop('sources')
        return dataset_config
    
    # 2. 独立格式: [general] + [[datasets]]
    if 'datasets' in config:
        # 如果有 [general] 块，合并到顶层
        if 'general' in config:
            config.update(config['general'])
        return config
    
    # 3. 根级别配置 (兼容旧版)
    return config


def create_reg_dataloader(args) -> Optional[DataLoader]:
    """
    创建正则数据集的 DataLoader（用于防止过拟合）。
    
    Args:
        args: 训练参数，包含 dataset_config
        
    Returns:
        DataLoader: 正则数据加载器，如果未启用则返回 None
    """
    # 读取配置
    if not hasattr(args, 'dataset_config') or not args.dataset_config:
        return None
    
    reg_config = _read_reg_dataset_config(args.dataset_config)
    
    if not reg_config.get('enabled', False):
        return None
    
    datasets = reg_config.get('datasets', [])
    if not datasets:
        logger.info("正则数据集已启用但未配置数据源，跳过")
        return None
    
    batch_size = getattr(args, 'batch_size', 4)
    num_workers = getattr(args, 'num_workers', 4)
    max_sequence_length = getattr(args, 'max_sequence_length', 512)
    cache_arch = getattr(args, 'cache_arch', 'zi')
    
    logger.info(f"🛡️ 加载正则数据集 (防过拟合)，数据源: {len(datasets)} 个")
    
    # 创建 dataset
    dataset = ZImageLatentDataset(
        datasets=datasets,
        max_sequence_length=max_sequence_length,
        cache_arch=cache_arch,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    
    logger.info(f"  正则数据集大小: {len(dataset)} 样本, {len(dataloader)} batches")
    logger.info(f"  正则权重: {reg_config.get('weight', 1.0)}, 混合比例: {reg_config.get('ratio', 0.5)}")
    
    return dataloader


def _read_reg_dataset_config(config_path: str) -> dict:
    """
    读取正则数据集配置 [reg_dataset]
    """
    if toml is None:
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = toml.load(f)
    
    if 'reg_dataset' in config:
        reg_config = config['reg_dataset'].copy()
        # 将 sources 重命名为 datasets
        if 'sources' in reg_config:
            reg_config['datasets'] = reg_config.pop('sources')
        return reg_config
    
    return {}


def get_reg_config(args) -> dict:
    """
    获取正则数据集配置参数（weight, ratio）供训练脚本使用
    """
    if not hasattr(args, 'dataset_config') or not args.dataset_config:
        return {'enabled': False, 'weight': 1.0, 'ratio': 0.5}
    
    return _read_reg_dataset_config(args.dataset_config)