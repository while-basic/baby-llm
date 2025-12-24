#!/bin/bash
#----------------------------------------------------------------------------
#File:       run.sh
#Project:     NeuralChild
#Created by:  Celaya Solutions, 2025
#Author:      Christopher Celaya <chris@chriscelaya.com>
#Description: Simple wrapper script to run NeuralChild CLI
#Version:     1.0.0
#License:     MIT
#Last Update: November 2025
#----------------------------------------------------------------------------

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Run the CLI with all arguments passed through
python3 neuralchild/cli.py "$@"

