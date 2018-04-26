for d in 100; do
    for m in n t d c; do
	python train.py -m $m -d $d -b 1000 -e 3 -g 0 > log/0417_speed_m$m
    done
done
