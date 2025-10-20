#!/bin/bash
cd ~/ultrathink-pilot
source .venv/bin/activate
nohup python train_exp1_strong.py > /tmp/exp1.log 2>&1 &
echo $! > /tmp/exp1.pid
echo "Exp1 launched with PID $(cat /tmp/exp1.pid)"
