#----------------------------------------------------------------------------
#File:       __init__.py
#Project:    Baby LLM - Unified Neural Child Development System
#Created by: Celaya Solutions, 2025
#Author:     Christopher Celaya <chris@chriscelaya.com>
#Description: Package initializer for the web module.
#Version:    1.0.0
#License:    MIT
#Last Update: January 2025
#----------------------------------------------------------------------------

"""Package initializer for the web module."""

from neural_child.web.app import create_app

__all__ = ['create_app']

