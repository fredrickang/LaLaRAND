i=1

while [ $i -le 10 ]
do
    echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 2 --algo 0 --hiding 0 --list taskset_0.9.txt --log_path Exp/PR/ --start 4 --end 5
    
    mv Exp/PR/ ../Hiding/mrr_10_ratio_2/Exp$i
    sh init.sh
    i=$(( i+1))
done

i=1

while [ $i -le 10 ]
do
    echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 2 --algo 1 --hiding 0 --list taskset_0.9.txt --log_path Exp/PR_LaLa/ --start 4 --end 5
    
    mv Exp/PR_LaLa/ ../Hiding/mrr_10_algo_limit_2/Exp$i
    sh init.sh
    i=$(( i+1))
done

