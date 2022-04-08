from fastai.vision.all import *
from fastai.distributed import *
from fastai.vision.models.xresnet import *
import argparse

# Cell
def rank0_first(func, *args, **kwargs):
    "Execute `func` in the Rank-0 process first, then in other ranks in parallel."
    if args: func = partial(func,*args)
    dummy_l = Learner(DataLoaders(device='cpu'), nn.Linear(1,1), loss_func=lambda: 0)
    with dummy_l.distrib_ctx(cuda_id=int(kwargs['cuda_id'])):
        if not rank_distrib(): res = func()
        distrib_barrier()
        if rank_distrib(): res = func()
    return res
    
  
@call_parse
def main(
    args:Param("Args to pass to script", nargs='...', opt=False)=''
):  
   
    os.environ["NCCL_SOCKET_IFNAME"]=args[1]
    
    path = rank0_first(untar_data, URLs.IMAGEWOOF_320,cuda_id=int(args[0]))
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        splitter=GrandparentSplitter(valid_name='val'),
        get_items=get_image_files, get_y=parent_label,
        item_tfms=[RandomResizedCrop(160), FlipItem(0.5)],
        batch_tfms=Normalize.from_stats(*imagenet_stats)
    ).dataloaders(path, path=path, bs=64)

    learn = Learner(dls, xresnet50(n_out=10), metrics=[accuracy,top_k_accuracy]).to_fp16()
    with learn.distrib_ctx(cuda_id=int(args[0])): learn.fit_flat_cos(2, 1e-3, cbs=MixUp(0.1))
