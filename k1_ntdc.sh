gpu=-1
k=1
d=100
w=0.0001
l=0.1
s=10
p=1
for model in n t d c; do
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
    -f k1 \
    > log/0404_m$model\c$kg\d$dim\k$k\s$s\w$weight\l$l\p$p
  done
done
