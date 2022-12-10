for count in {1..5}
do  
    echo "starting with file $count"
    python3 RL_learning.py --data $count
done