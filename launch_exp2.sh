#!/bin/bash
cd ~/ultrathink-pilot
source .venv/bin/activate
nohup python train_exp2_exp.py > /tmp/exp2.log 2>&1 &
echo $! > /tmp/exp2.pid
echo "Exp2 launched with PID $(cat /tmp/exp2.pid)"
