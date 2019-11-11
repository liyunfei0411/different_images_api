#!/bin/bash
find /home/yq/different_images_api/results/ -mtime +3 -type f -name \*.jpg | xargs rm -f
