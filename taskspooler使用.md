#### Task spooler
Usage: http://vicerveza.homeunix.net/~viric/soft/ts/

下载地址：https://codeload.github.com/xenogenesi/task-spooler/zip/master

#### 代码示例
$cat run_batch.py

```
#!/usr/bin/python
import os
import subprocess

def run_bash(cmd):
  p = subprocess.Popen(
  cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
  out = p.stdout.read().strip()
return out  # This is the stdout from the shell command

max_job_num_per_gpu = 1
os.system('ts -S %d' % max_job_num_per_gpu)

CUDA_VISIBLE_DEVICES = ['0', '1']
epochs = [5, 10, 20, 30, 40]

os.system('export BERT_BASE_DIR=../../data/chinese_L-12_H-768_A-12')
os.system('export XNLI_DIR=../../data/tiangong/slot_preprocess/video')

BERT_BASE_DIR = '../../data/uncased_L-12_H-768_A-12'
XNLI_DIR = '../../data/nlu_from_goo/snips'

count = 0
for epoch in epochs:
  device = CUDA_VISIBLE_DEVICES[count % len(CUDA_VISIBLE_DEVICES)]
  job_cmd = 'python run_classifier.py
    --task_name=slot
    --do_train=true
    --do_eval=true
    --do_predict=true
    --data_dir={}
    --vocab_file={}/vocab.txt
    --bert_config_file={}/bert_config.json
    --init_checkpoint={}/bert_model.ckpt
    --max_seq_length=50
    --train_batch_size=128
    --learning_rate=4e-5
    --num_train_epochs={}
    --output_dir=./output_epoch{}/ > log_training_epoch{}.txt 2>&1 '.format(XNLI_DIR, BERT_BASE_DIR, BERT_BASE_DIR, BERT_BASE_DIR, epoch, epoch, epoch)
  print(job_cmd)
  submit_cmd = "CUDA_VISIBLE_DEVICES={} TS_SOCKET=/tmp/socket-ts.gpu_queue_{} ts bash -c '{}'".format(device, device, job_cmd)
  run_bash(submit_cmd)
  count += 1
```
