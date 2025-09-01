# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import multiprocessing
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional

import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


def get_custom_reward_fn(config: DictConfig) -> Optional[RawRewardFn]:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        config (dict): Configuration dictionary containing custom_reward_function
                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    assert function_name is not None
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    return partial(_call_with_kwargs, raw_fn, reward_kwargs)


def load_reward_manager(
    config: DictConfig, tokenizer: Any, num_examine: int, **reward_kwargs: Any
) -> AbstractRewardManager:
    """
    Load and initialize a reward manager based on the configuration.

    Args:
        config: PPO trainer configuration object containing reward_model fields.
        tokenizer: Tokenizer object used for processing text.
        num_examine: Number of samples to examine.
        **reward_kwargs: Additional keyword arguments for the reward manager.

    Returns:
        An instance of the specified reward manager class.
    """

    # The list of pre-defined reward managers are defined in `verl/workers/reward_manager/`:
    # naive: NaiveRewardManager
    # prime: PrimeRewardManager
    # batch: BatchRewardManager
    # dapo: DAPORewardManager
    # Note(haibin.lin): For custom reward managers, please make sure they are imported and
    # registered via `verl.workers.reward_manager.register`
    # By default reward_manager is set to naive (NaiveRewardManager)
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)

    # Try to get a custom reward function based on the configuration
    compute_score = get_custom_reward_fn(config)
    final_compute_score = compute_score

    if compute_score is None:
        sandbox_config = config.reward_model.get("sandbox_fusion")
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024)
        if sandbox_url:
            sandbox_manager = multiprocessing.Manager()
            # Create a semaphore to control concurrent access to the sandbox
            _concurrent_semaphore = sandbox_manager.Semaphore(sandbox_config.get("max_concurrent", 64))
            final_compute_score = partial(
                default_compute_score,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score

    # Instantiate and return the reward manager with the specified parameters
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


def compute_reward_with_separation(data: DataProto, reward_fn: AbstractRewardManager) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute separated rewards for translate and think sections.
    
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
        
    Returns:
        Tuple of reward tensor and extra info dictionary with separated rewards.
    """
    try:
        # Try to call reward function with separation enabled
        try:
            # Call with return_separated=True
            reward_result = reward_fn(data, return_dict=True, return_separated=True)
        except TypeError:
            # If reward_fn doesn't support return_separated, fallback to standard computation
            print("Warning: reward_fn doesn't support return_separated parameter, falling back to standard computation")
            reward_result = reward_fn(data, return_dict=True)
            
        if isinstance(reward_result, dict) and "reward_separated" in reward_result:
            # Separated rewards are available
            separated_rewards = reward_result["reward_separated"]
            
            # Create token-level rewards for each section
            batch_size, seq_len = data.batch["input_ids"].shape
            device = data.batch["input_ids"].device
            
            # Initialize separated reward tensors
            R_tr_tensor = torch.zeros(batch_size, seq_len, device=device)
            R_th_tensor = torch.zeros(batch_size, seq_len, device=device)
            
            # Fill rewards (assuming they are outcome rewards - same value for all tokens in response)
            response_mask = data.batch.get("response_mask", torch.ones(batch_size, seq_len, device=device))
            
            for i in range(batch_size):
                if isinstance(separated_rewards["R_tr"], list):
                    R_tr_value = separated_rewards["R_tr"][i] if i < len(separated_rewards["R_tr"]) else 0.0
                    R_th_value = separated_rewards["R_th"][i] if i < len(separated_rewards["R_th"]) else 0.0
                else:
                    R_tr_value = separated_rewards["R_tr"]
                    R_th_value = separated_rewards["R_th"]
                
                # Set reward for all response tokens (GRPO style)
                R_tr_tensor[i] = R_tr_value * response_mask[i]
                R_th_tensor[i] = R_th_value * response_mask[i]
            
            # Create combined reward for backward compatibility
            combined_reward = R_tr_tensor + R_th_tensor
            
            # Create separated masks if possible
            response_mask_tr = None
            response_mask_th = None
            
            try:
                # Try to create separated masks using tokenizer from reward_fn
                if hasattr(reward_fn, 'tokenizer') and reward_fn.tokenizer is not None:
                    from verl.trainer.ppo.core_algos import create_separated_masks
                    responses = data.batch.get("responses", data.batch.get("input_ids"))
                    response_mask_tr, response_mask_th = create_separated_masks(
                        responses, reward_fn.tokenizer, response_mask
                    )
                    print(f"Created separated masks - TR: {response_mask_tr.sum().item()} tokens, TH: {response_mask_th.sum().item()} tokens")
            except Exception as e:
                print(f"Warning: Could not create separated masks: {e}")
            
            # Create reward tensor with separated info
            batch_data = {
                "token_level_rewards": combined_reward,
                "token_level_scores": combined_reward,  # alias
                "token_level_rewards_separated": {
                    "R_tr": R_tr_tensor,
                    "R_th": R_th_tensor,
                    "R_combined": combined_reward
                }
            }
            
            # Add separated masks if available
            if response_mask_tr is not None and response_mask_th is not None:
                batch_data["response_mask_tr"] = response_mask_tr
                batch_data["response_mask_th"] = response_mask_th
            
            reward_tensor = DataProto(batch=batch_data)
            
            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
            reward_extra_infos_dict.update({
                "separated_rewards_available": True,
                "avg_R_tr": float(torch.mean(R_tr_tensor[response_mask.bool()]).item()) if response_mask.sum() > 0 else 0.0,
                "avg_R_th": float(torch.mean(R_th_tensor[response_mask.bool()]).item()) if response_mask.sum() > 0 else 0.0,
            })
            
        else:
            # No separated rewards available, fall back to standard computation
            print("Warning: Separated rewards not available from reward function, falling back to standard computation")
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
            
    except Exception as e:
        print(f"Error in compute_reward_with_separation: {e}")
        # Fall back to standard reward computation
        reward_tensor, reward_extra_infos_dict = compute_reward(data, reward_fn)
        
    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config=None, tokenizer=None, reward_fn=None):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    if reward_fn is None:
        assert config is not None and tokenizer is not None, (
            "config and tokenizer must not be None when reward_fn is None"
        )

        warnings.warn("using config and tokenizer with compute_reward_async is deprecated", stacklevel=2)
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    return compute_reward(data, reward_fn)
