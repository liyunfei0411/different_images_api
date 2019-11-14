#!/bin/bash
python mkdir.py && gunicorn -c gunicorn.conf identifying_objects:app

