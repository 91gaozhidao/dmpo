# MIT License
#
# Copyright (c) 2024 Intelligent Robot Motion Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Legacy Google Drive download URL helpers.

These functions provide fallback URL lookup for datasets, normalization files,
and checkpoints when the preferred Hugging Face download path (hf://) is not
used and the local file does not exist.

Preferred path: use ``hf://`` prefixes in config files so that
``util.hf_download`` handles resolution automatically.  The functions below
are retained only for backward compatibility with older configs that rely on
Google Drive mirrors.
"""

import logging

from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Google Drive folder / file ID mappings (legacy)
# ---------------------------------------------------------------------------
# Add concrete mappings here if Google Drive mirrors are set up.
# Format: env_name -> Google Drive URL or file ID.
_DATASET_URLS: dict = {}
_NORMALIZATION_URLS: dict = {}
_CHECKPOINT_URLS: dict = {}


def get_dataset_download_url(cfg) -> Optional[str]:
    """Return a Google Drive download URL for the training dataset.

    Falls back to ``None`` when no mapping is available for the given config.
    Callers should prefer ``hf://`` paths in config files instead of relying
    on this legacy helper.
    """
    env_name = cfg.get("env_name", None)
    url = _DATASET_URLS.get(env_name, None)
    if url is None:
        log.warning(
            "No legacy Google Drive URL for dataset (env_name=%s). "
            "Use an hf:// path or place the file locally.",
            env_name,
        )
    return url


def get_normalization_download_url(cfg) -> Optional[str]:
    """Return a Google Drive download URL for normalization statistics.

    Falls back to ``None`` when no mapping is available for the given config.
    Callers should prefer ``hf://`` paths in config files instead of relying
    on this legacy helper.
    """
    env_name = cfg.get("env_name", None)
    url = _NORMALIZATION_URLS.get(env_name, None)
    if url is None:
        log.warning(
            "No legacy Google Drive URL for normalization (env_name=%s). "
            "Use an hf:// path or place the file locally.",
            env_name,
        )
    return url


def get_checkpoint_download_url(cfg) -> Optional[str]:
    """Return a Google Drive download URL for a pre-trained checkpoint.

    Falls back to ``None`` when no mapping is available for the given config.
    Callers should prefer ``hf://`` paths in config files instead of relying
    on this legacy helper.
    """
    env_name = cfg.get("env_name", None)
    url = _CHECKPOINT_URLS.get(env_name, None)
    if url is None:
        log.warning(
            "No legacy Google Drive URL for checkpoint (env_name=%s). "
            "Use an hf:// path or place the file locally.",
            env_name,
        )
    return url
