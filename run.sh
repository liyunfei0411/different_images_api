#!/bin/bash
python mkdir.py && gunicorn -w 6 -b 0.0.0.0:5001 identifying_objects:app

