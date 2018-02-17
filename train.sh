for kg in w f; do
    for m in t d c; do
	for k in 1 4; do
            python train.py -m $m -v 0 -k $k -g 0 > log/0121_no_matpro_$kg\_m$m\_k$k
	done
    done
done
