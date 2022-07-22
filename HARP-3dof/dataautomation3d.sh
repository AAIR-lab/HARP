#!/bin/sh

for i in `seq 1 100`
do
    python simulation.py envs3d/29.0.xml 29.0 $i
done