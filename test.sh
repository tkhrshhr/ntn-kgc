# Receive experiment folder name as the first argument
exp_path=trained_model/$1
mkdir -p test_result/$1
for file in `\find $exp_path -maxdepth 1 -type f`; do
    file=$(basename $file)
    python test.py -f $file -cr class
done
