rm input/raw/test/*
# rm input/raw/test-labels/*
rm input/waypoint-3d/test*

# declare -a arr=("5.2.1" "8.0.1" "8.0.2" "8.0.3" "8.0.4" "10.2.1" "10.2.2" "18.1.1")
declare -a arr=("20.0.1" "21.0.1" "25.0.1" "27.0.1")

# loop through the envs
for i in "${arr[@]}"
do
    echo "$i"

    rm input/raw/test/*
    # rm input/raw/test-labels/*
    rm input/waypoint-3d/test*

    cp input/raw/samples_01/$i.npy input/raw/test/$i.npy
    # cp input/raw/samples/8.0.1_lbl.npy input/raw/test-labels/8.0.1_lbl.npy

    python tfrecorder.py test
    python test.py $i
done