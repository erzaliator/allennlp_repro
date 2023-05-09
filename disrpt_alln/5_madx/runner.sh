#history: it was first used to run same_pre1_v5, then same_pre. then same_pre was modified for learning rate decay and run again.

nohup python 01_madx_samepre.py --lang1_index=0 --cuda=4 > logs_runtime/runtime_0.out 2> logs_runtime/runtime_0.err &
# nohup python 01_madx_samepre.py --lang1_index=1 --cuda=5 > logs_runtime/runtime_1.out 2> logs_runtime/runtime_1.err &
# nohup python 01_madx_samepre.py --lang1_index=2 --cuda=4 > logs_runtime/runtime_2.out 2> logs_runtime/runtime_2.err &
# nohup python 01_madx_samepre.py --lang1_index=3 --cuda=5 > logs_runtime/runtime_3.out 2> logs_runtime/runtime_3.err &
# nohup python 01_madx_samepre.py --lang1_index=4 --cuda=7 > logs_runtime/runtime_4.out 2> logs_runtime/runtime_4.err &
# nohup python 01_madx_samepre.py --lang1_index=5 --cuda=4 > logs_runtime/runtime_5.out 2> logs_runtime/runtime_5.err &
# nohup python 01_madx_samepre.py --lang1_index=6 --cuda=4 > logs_runtime/runtime_6.out 2> logs_runtime/runtime_6.err &
# nohup python 01_madx_samepre.py --lang1_index=7 --cuda=1 > logs_runtime/runtime_7.out 2> logs_runtime/runtime_7.err &
# nohup python 01_madx_samepre.py --lang1_index=8 --cuda=6 > logs_runtime/runtime_8.out 2> logs_runtime/runtime_8.err &
# nohup python 01_madx_samepre.py --lang1_index=9 --cuda=7 > logs_runtime/runtime_9.out 2> logs_runtime/runtime_9.err &
# nohup python 01_madx_samepre.py --lang1_index=10 --cuda=1 > logs_runtime/runtime_10.out 2> logs_runtime/runtime_10.err &
# nohup python 01_madx_samepre.py --lang1_index=11 --cuda=5 > logs_runtime/runtime_11.out 2> logs_runtime/runtime_11.err &
# nohup python 01_madx_samepre.py --lang1_index=12 --cuda=6 > logs_runtime/runtime_12.out 2> logs_runtime/runtime_12.err &
# nohup python 01_madx_samepre.py --lang1_index=13 --cuda=7 > logs_runtime/runtime_13.out 2> logs_runtime/runtime_13.err &