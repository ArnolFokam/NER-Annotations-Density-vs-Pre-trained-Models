touch datastore2
# v0150_1-a v0150_1-b v0150_1-c v0150_1-d v0150_1-e v0150_1-f v0150_1-g v0150_1-h v0150_1-i v0150_1-j v0150_1-k v0150_1-l v0151_4-a v0151_4-b v0151_4-c v0151_4-d v0148-a v0148-b v0148-c 
for i in `cat all_exps`; do 
    NUM_JOBS=`squeue | grep mfokam | grep bigbatch | wc -l`
    N=20
    if ! grep -q $i datastore2; then
        # echo "Not found $i. Possibly Running"
        if [ "$NUM_JOBS" -lt "$N" ]; then
            echo "$NUM_JOBS JOBS LESS $N, will definitely run $i"
            ./run.sh $i
            # sbatch --exclude=mscluster33,mscluster22,mscluster71,mscluster63,mscluster94,mscluster30 artifacts/slurms/reeval/220731/$i
            echo $i >> datastore2
            echo "Ran $i"
        else
            echo "$NUM_JOBS bigger $N. Will not run $i"
            break
        fi

    fi
done
