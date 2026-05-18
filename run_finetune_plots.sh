#!/bin/bash
cd /cs/student/projects1/2023/muhamaaz/year-long
source crowdnav-env/bin/activate
python -u generate_finetune_plots.py --ft-only --save-json results_ft.json > logs/finetune_plots.log 2>&1
