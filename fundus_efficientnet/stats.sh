grep -Eo "Acc@1 [0-9]{1,2}.[0-9]{3}" log_4.txt 
grep "Epoch: .*loss" log_4.txt |grep -Eo "loss: [0-9]{1,3}.[0-9]{1,4}"
grep "Epoch: .*loss" log_4.txt |grep -Eo "lr: .*img" 
grep -Eo "Acc@2 [0-9]{1,2}.[0-9]{3}" log_4.txt 
grep "Epoch:" log_4.txt |grep -Eo "acc1: [0-9]{1,3}.[0-9]{3}" 