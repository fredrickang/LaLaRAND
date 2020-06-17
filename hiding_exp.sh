i=1

while [ $i -le 10 ]
do
    echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 4 --algo 0 --hiding 0 --list new_taskset.txt --log_path Exp/DART_GC/ --end 1
        
    mv Exp/DART_GC/ ../Response/DART_GC/Exp$i
    sh init.sh
    i=$(( i+1 ))
done

#i=1

#while [ $i -le 10 ]
#do
#    echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 2 --algo 1 --hiding 0 --list new_taskset.txt --log_path Exp/PR_LaLa/ --end 1
#    
#    mv Exp/PR_LaLa/ ../Response/PR_LaLa/Exp$i
#    sh init.sh
#    i=$(( i+1))
#done

