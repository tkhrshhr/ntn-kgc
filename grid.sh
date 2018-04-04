model="t"
p=1
gpu=2
for kg in f; do
  for dim in 100; do
    for k in 1 4; do
      for s in 2 10; do
        for weight in 0.001 0.0001 0.00001; do
          for l in 0.1 0.05 0.01; do
            python train.py \
            -e 300 \
            -sp 50 \
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
            > log/0329_e300m$model\g$gpu\c$kg\d$dim\k$k\s$s\w$weight\l$l\p$p
          done
        done
      done
    done
  done
done
