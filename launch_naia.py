import subprocess,torch,os,sys
from fastcore.basics import *
from fastcore.script import *

@call_parse
def main(
    gpus_to_use:Param("The ids of GPUs to use for distributed training in this node. Example: 0,1", str)='all',
    total_gpus:Param("The total number of GPUs to use for distributed training in all nodes", int)=1,
    nr:Param("The ranking of the GPUs in this node. Example: 0,1", str)='0',
    ethernet_adapter:Param("Ethernet Adapter to use for this node", str)='',
    master_IP:Param("The IP of the master in this DDP", str)='',
    master_PORT:Param("The PORT of the master in this DDP", str)='',
    script:Param("Script to run", str, opt=False)='',
    args:Param("Args to pass to script", nargs='...', opt=False)=''
):
    "PyTorch distributed training launch helper that spawns multiple distributed processes"
    current_env = os.environ.copy()
    gpus_to_use_list = list(range(torch.cuda.device_count())) if gpus_to_use=='all' else gpus_to_use.split(',')
    gpus_node = torch.cuda.device_count() if gpus_to_use=='all' else len(gpus_to_use.split(','))
    rankings =  nr.split(',')
    current_env["WORLD_SIZE"] = str(total_gpus)
    current_env["MASTER_ADDR"] = master_IP
    current_env["MASTER_PORT"] = master_PORT
    
    if(gpus_node!=len(rankings)):
        print(f'ERROR: The number of GPUs to use in this node ({gpus_node}) does not match with the number of ranking ids introduced (--nr={rankings})')
        exit(-1)
    procs = []
    for i,gpu in enumerate(gpus_to_use_list):
        current_env["RANK"],current_env["DEFAULT_GPU"] = str(rankings[i]),str(gpu)
        procs.append(subprocess.Popen([sys.executable, "-u", script] + list(str(gpu))+ethernet_adapter.split(' '), env=current_env))
    for p in procs: p.wait()
