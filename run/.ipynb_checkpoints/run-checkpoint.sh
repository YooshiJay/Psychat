#!/bin/bash
service nginx reload
service nginx restart
jupyter nbconvert --to script ../Qwen1.5_14B_Psychat_Predict_webui.ipynb --output-dir=./
nohup python Qwen1.5_14B_Psychat_Predict_webui.py &
