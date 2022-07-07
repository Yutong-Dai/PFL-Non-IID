# python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m cnn -algo FedROD -gr 100 -did 0 -go reproduce --local_steps 5 --local_learning_rate 0.01
# nohup python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m dnn -algo FedROD -gr 100 -did 0 -go reproduce > Cifar100_FedROD.out 2>&1


# nohup python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m cnn -algo FedROD -gr 1000 -did 0 -go reproduce --local_steps 5 --local_learning_rate 0.01> Cifar100_FedROD.out 2>&1 &

# nohup python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m cnn -algo FedAvg -gr 1000 -did 0 -go reproduce --local_steps 5 --local_learning_rate 0.01> Cifar100_FedAvg.out 2>&1 &

# nohup python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m cnn -algo pFedMe -gr 100 -did 1 -go reproduce --local_steps 5 --local_learning_rate 0.01> Cifar100_pFedMe.out 2>&1 &

# nohup python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m cnn -algo FedPer -gr 100 -did 1 -go reproduce --local_steps 5 --local_learning_rate 0.01> Cifar100_FedPer.out 2>&1 &


# nohup python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m cnn -algo FedAvg -gr 100 -did 0 -go reproduce --local_steps 5 --local_learning_rate 0.01> Cifar100_FedAvg.out 2>&1 &


python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m cnn -algo FedPer -gr 100 -did 1 -go reproduce --local_steps 5 --local_learning_rate 0.01
python -u main.py -lbs 64 -nc 20 -jr 0.4 -nb 100 -data Cifar100 -m cnn -algo FedAvg -gr 100 -did 1 -go reproduce --local_steps 5 --local_learning_rate 0.01