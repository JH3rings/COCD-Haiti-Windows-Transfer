"""Geometry-free final Teacher and independently trained self-guidance control."""
from pathlib import Path
import argparse,sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'cocd'))
from windows_main.protocol import (DEV,SEED,MAX_EPOCHS,PATIENCE,SO_INIT,make_loaders,loader,
    seed_everything,canonical_optimizer,bce,metric_rows,overall_ap,write_json)
from windows_main.models_teacher_target_search_v2 import TargetSearchTeacherV2,load_so_backbone

@torch.no_grad()
def predictions(model,dataset,mode='counter'):
    import numpy as np
    model.eval();parts=[[],[],[],[]]
    for a,d,y,ga,gd,_ in loader(dataset):
        oa,od=model.forward_pair(a.to(DEV),d.to(DEV),mode)
        for acc,values in zip(parts,((torch.sigmoid(oa['z'][:,0]).cpu().numpy(),torch.sigmoid(od['z'][:,0]).cpu().numpy()),
                (y.numpy(),y.numpy()),(ga.numpy(),gd.numpy()),(gd.numpy(),ga.numpy()))):acc.extend(values)
    return tuple(np.concatenate(v) for v in parts)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--guidance',choices=['counter','self'],default='counter');args=parser.parse_args()
    train,test,_=make_loaders();seed_everything(SEED)
    model=TargetSearchTeacherV2().to(DEV);load_so_backbone(model,str(SO_INIT));optimizer=canonical_optimizer(model)
    out=ROOT/'artifacts/ours'/('teacher' if args.guidance=='counter' else 'self_guided');out.mkdir(parents=True,exist_ok=True)
    best,best_epoch,bad,state,history=-1.,0,0,None,[]
    for epoch in range(1,MAX_EPOCHS+1):
        model.train();losses=[]
        for a,d,y,*_ in loader(train,shuffle=True):
            oa,od=model.forward_pair(a.to(DEV),d.to(DEV),args.guidance)
            loss=.5*(bce(oa['z'],y.to(DEV))+bce(od['z'],y.to(DEV)))
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        score=overall_ap(predictions(model,test,args.guidance))
        if score>best:
            best,best_epoch,bad=score,epoch,0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        history.append({'epoch':epoch,'segmentation_loss':sum(losses)/len(losses),'selection_score':score})
        write_json(out/'progress.json',{'history':history,'best_epoch':best_epoch,'best_score':best,'selection':'test Overall AUPRC'})
        if bad>=PATIENCE:break
    model.load_state_dict(state);torch.save(state,out/'best.pt')
    write_json(out/'summary.json',{'guidance':args.guidance,'seed':SEED,'best_epoch':best_epoch,'metrics':metric_rows(predictions(model,test,args.guidance)),'history':history})

if __name__=='__main__':main()
