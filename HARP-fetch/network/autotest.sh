# rm input/raw/test/*
# rm input/raw/test-labels/*
# rm input/waypoint-3d/test*

# declare -a arr=("5.2.1" "8.0.1" "8.0.2" "8.0.3" "8.0.4" "10.2.1" "10.2.2" "18.1.1")
# declare -a arr=("test_env1" "test_env2" "test_env3" "test_env4" "test_env5" "4.0.1" "11.0.1" "13.0.1")

declare -a arr = ("29.0")

# loop through the envs
for i in "${arr[@]}"
do
    echo "$i"

    # rm input/raw/test/*
    # rm input/raw/test-labels/*
    # rm input/se2-pd/test*

    # cp input/raw/samples/$i.npy input/raw/test/$i.npy
    # cp input/raw/samples/8.0.1_lbl.npy input/raw/test-labels/8.0.1_lbl.npy

    python tfrecorder.py test
    python test.py $i
done