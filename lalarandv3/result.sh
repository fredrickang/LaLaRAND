sudo ./lalarand -sync 2

if [ $? -eq 1 ]
then
    echo "Success"
else
    echo "Fail" >&2
fi
