<< COMM
model=("n" "t" "d" "c")
declare -A gpu_ids
gpu_ids["n"]=0
gpu_ids["t"]=1
gpu_ids["d"]=2
gpu_ids["c"]=3

echo ${gpu_ids["n"]}
for model in ${model[@]}; do
  echo ${gpu_ids[${model}]}
done
COMM

model=$1
kg_array=("w11" "f13")
declare -A gpu_ids
gpu_ids["n"]=0
gpu_ids["t"]=1
gpu_ids["d"]=2
gpu_ids["c"]=3

e=500
d_array=(25 50 100 150 200)
s=10
p=1
sp=50
k_array=(4)
l_array=(0.2 0.15 0.1 0.05)
w_array=(0.0005 0.0001 0.00005)
b_array=(50 100 200 500)
for model in $model; do
    for kg in ${kg_array[@]}; do
	for d in ${d_array[@]}; do
	    for k in ${k_array[@]}; do
		for l in ${l_array[@]}; do
		    for w in ${w_array[@]}; do
			for b in ${b_array[@]}; do      
			    python train.py \
			    -sp $sp \
			    -c $kg \
			    -m $model \
			    -e $e \
			    -g ${gpu_ids[${model}]} \
			    -d $d \
			    -k $k \
			    -s $s \
			    -w $w \
			    -l $l \
			    -p $p \
			    -b $b \
			    -f w11f13 \
			    > log/0503_m$model\c$kg\d$dim\k$k\s$s\w$w\l$l\p$p\b$b
			done
		    done
		done
	    done
	done
    done
done
