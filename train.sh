model="t"
gpu=2
p=1
for kg in f; do
  for dim in 100; do
    for k in 1 4; do
      for s in 2 10; do
        for weight in 0.001 0.0001 0.00001; do
          for l in 0.1 0.05 0.01; do
          python train.py \
          -e 300 \
          -m $model \
          -g $gpu \
          -c $kg \
          -d $dim \
          -k $k \
          -s $s \
          -w $weight \
          -l $l \
          -p $p \
          -f grid \
          > log/0328_e300_m$model\g$gpu\c$kg\d$dim\k$k\s$s\w$weight\l$l\p$p
          done
        done
      done
    done
  done
done
