for kg in w f; do
    for m in t d c; do
        python train.py -m $m -v 0 -g 0 > log/0121_no_matpro_$kg\_m$m
    done
done
