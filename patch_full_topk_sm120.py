#!/usr/bin/env python3
"""Compatibility entry point: FullTopK is now part of the complete bundle.

Copy patch_sm120.py and patches/ alongside this file. Repeated execution verifies
the installed bundle, including the retained SM12x Triton paged-decode route.
"""

from patch_sm120 import main


if __name__ == "__main__":
    main()
