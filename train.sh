model="n"
gpu=-1
for kg in w f; do
  for dim in 100 150 200; do
    for weight in 0.01 0.001 0.0001; do
      for k in 1 4 10; do
        for l in 0.1 0.05; do
          python train.py -c $kg -m $model -g $gpu -d $dim -w $weight -k $k -l $l > log/0218_c$kg\m$model\g$gpu\d$dim\w$weight\k$k\l$l
        done
      done
    done
  done
done
