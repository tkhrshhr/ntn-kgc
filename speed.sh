for d in 8 50 100 150 200 250 300 350 400 450 500; do
    for m in n t d c; do
	python train.py -m $m -d $d -b 1000 -e 3 -g 1 > log/m$m\d$d
    done
    for p in 1 2 3 10 25; do
	python train.py -m s -d $d -b 1000 -e 3 -p $p -g 1 > log/s\d$d\p$p
    done
done
