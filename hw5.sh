#!/bin/bash

gdown --id '1VvQonwOyuk1lTaJOoUx-Vdy6xPBfC80z' --output model.torch
gdown --id '1mfs3oSc0L_CaV8fedLhh5BHQhBq2FRRT' --output model_confusion.torch

python3 -W ignore gen_image.py $1 $2
