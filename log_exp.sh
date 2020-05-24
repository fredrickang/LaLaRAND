i=0
while [ $i -le 4 ]
do
#echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 1 --algo 0 --list taskset_0.9.txt 
#echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 1 --algo 1 --list taskset_0.9.txt --log_path Exp/ALL_LaLa/
    echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 2 --algo 0 --list taskset_0.9.txt --log_path Exp/PR/  
    echo 'Kagnws0906' | sudo -S python3 experiment.py --baseline 2 --algo 1 --list taskset_0.9.txt --log_path Exp/PR_LaLa/ 
    mv Exp ../LIMIT/EXP_LOG$i
    mkdir Exp
    sh init.sh

    i=$(( i+1 ))
done



