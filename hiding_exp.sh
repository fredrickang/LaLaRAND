i=1

while [ $i -le 10 ]
do
    echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 2 --algo 0 --hiding 0 --list taskset_0.9.txt --log_path Exp/PR/ --start 1 --end 2

    mv Exp/PR ../Hiding/max_response/Exp$i
    sh init.sh
    i=$(( i+1))
done
