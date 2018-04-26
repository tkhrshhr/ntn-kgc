model=s
gpu=1
k=4
d=100
w=0.0001
l=0.1
s=10
for p in 1 2 3 10 25; do
  for kg in w f; do
    if [ $kg = "w" ]; then
      e=300
    else
      e=100
    fi
    python train.py \
    -sp 50 \
    -c $kg \
    -m $model \
    -e $e \
    -g $gpu \
    -d $d \
    -k $k \
    -s $s \
    -w $w \
    -l $l \
    -p $p \
    -f k4 \
    > log/0404_m$model\c$kg\d$dim\k$k\s$s\w$w\l$l\p$p
  done
done

