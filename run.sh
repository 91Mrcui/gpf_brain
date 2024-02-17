command1=" python pre_train.py --epochs 1000 --lr 0.001 --batch_size 64 --g_num_layers 5"
command2=" python prompt_tuning.py --epochs 300 --lr 0.001 --batch_size 64"
# 执行命令1
echo "Executing command: $command1"
eval "$command1 >> record1.txt &"
wait
# 执行命令2
echo "Executing command: $command2"
eval "$command2 >> record2.txt &"
wait