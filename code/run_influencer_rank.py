#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point.

This file is intentionally tiny:
  - parse CLI
  - set CUDA_VISIBLE_DEVICES BEFORE torch import
  - import influencer_rank.app and run
"""

import os
from influencer_rank.cli import parse_args_pre_torch

def main():
    args = parse_args_pre_torch()
    # must be set before torch import (cli already did CUDA_VISIBLE_DEVICES)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    from influencer_rank.app import main as app_main
    return app_main(args)

if __name__ == "__main__":
    raise SystemExit(main())
