echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 1 --algo 0 --list taskset_0.9.txt 
echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 1 --algo 1 --list taskset_0.9.txt --log_path Exp/ALL_LaLa/
echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 2 --algo 0 --list taskset_0.9.txt --log_path Exp/PR/
echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 2 --algo 1 --list taskset_0.9.txt --log_path Exp/PR_LaLa/
echo 'Kangws0906' | sudo -S python3 experiment.py --baseline 3 --algo 0 --list taskset_0.9.txt 
