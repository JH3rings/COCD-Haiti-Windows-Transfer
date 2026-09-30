"""Matched current two-stage loss arms; isolated outputs, canonical stop-grad KD."""
from pathlib import Path
import sys,json,math,time,importlib.util
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/objectives'
sys.path.insert(0,str(ROOT/'cocd'))
from windows_main.protocol import *
from windows_main import train_student as canonical

def main():
    torch.set_num_threads(4)
    train,test,split_info=make_loaders()
    for arm,usep,usee in [('segmentation',False,False),('policy',True,False),('evidence',False,True)]:
        dest=OUT/arm;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'summary.json').exists():continue
        seed_everything();teacher=load_teacher();student=make_student();opt=canonical_optimizer(student)
        history=[];best=-np.inf;best_epoch=None;bad=0;pc=ec=lam=None
        protocol=protocol_record(arm,{'stage1_epochs':5,'calibration_epoch':5,'policy_enabled':usep,'evidence_enabled':usee,
            'component_coefficients':{'policy':.5 if usep else 0.,'evidence':.5 if usee else 0.},
            'ablation_control':'retain canonical per-component coefficient; do not reallocate disabled loss weight',
            'canonical_source':str(canonical.HERE),'geometry_use':'none in training; posthoc metrics only',
            'split':split_info,'execution_change':'omit diagnostic grad_norm calls, torch threads=4; same optimizer and losses'})
        begin=time.time()
        for epoch in range(1,MAX_EPOCHS+1):
            student.train();sums={k:0. for k in ('seg','policy','evidence','weighted_policy','weighted_evidence','total')};n=0
            for a,d,y,*_ in loader(train,shuffle=True):
                a,d,y=a.to(DEV),d.to(DEV),y.to(DEV)
                if epoch<=4 or (not usep and not usee):
                    oa,od=student(a),student(d);lp=le=oa['z'].new_zeros(())
                else:lp,le,oa,od=canonical.kd_terms(student,teacher,a,d)
                seg=.5*(bce(oa['z'],y)+bce(od['z'],y));wp=we=seg.new_zeros(())
                if epoch>=6:
                    multiplier=.25+.75*.5*(1+math.cos(math.pi*(epoch-6)/(MAX_EPOCHS-6)))
                    if usep:wp=.5*lam*multiplier*lp/max(pc,EPS)
                    if usee:we=.5*lam*multiplier*le/max(ec,EPS)
                loss=seg+wp+we;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                for k,v in [('seg',seg),('policy',lp),('evidence',le),('weighted_policy',wp),('weighted_evidence',we),('total',loss)]:sums[k]+=float(v.detach())
                n+=1
            means={k:v/n for k,v in sums.items()}
            if epoch==5:pc,ec=means['policy'],means['evidence'];lam=.25*means['seg']
            prediction=student_predictions(student,test);score=overall_ap(prediction)
            history.append({'epoch':epoch,**means,'test_overall_auprc':score,'seconds_since_start':time.time()-begin})
            if score>best:
                best=score;best_epoch=epoch;bad=0
                torch.save(student.state_dict(),dest/'best.pt')
                np.savez_compressed(dest/'best_predictions.npz',p=prediction[0],y=prediction[1],target_geometry=prediction[2],counter_geometry=prediction[3])
                write_json(dest/'best_metrics.json',metric_rows(prediction))
            else:bad+=1
            write_json(dest/'progress.json',{'status':'RUNNING','protocol':protocol,'best_epoch':best_epoch,'history':history})
            print(f'{arm} epoch={epoch} AP={score:.8f} best={best_epoch} elapsed={time.time()-begin:.1f}s',flush=True)
            if bad>=PATIENCE:break
        write_json(dest/'summary.json',{'status':'COMPLETE','stopping_reason':'patience 2' if bad>=PATIENCE else 'max epochs','protocol':protocol,'best_epoch':best_epoch,'metrics':json.loads((dest/'best_metrics.json').read_text()),'history':history,
            'prediction_order':'for each ordered batch of 16 locations: ASC rows then DESC rows; see matched_prediction_index.csv'})
        write_json(dest/'progress.json',{'status':'COMPLETE','best_epoch':best_epoch,'history':history,'protocol':protocol})
        del teacher,student,opt;torch.cuda.empty_cache()
    index=[]
    for start in range(0,len(test),BATCH):
        for orbit in ('asc','desc'):
            for sid in test.ids[start:start+BATCH]:index.append({'row':len(index),'sample_id':sid,'orbit':orbit})
    import csv
    with (OUT/'matched_prediction_index.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(index[0]));w.writeheader();w.writerows(index)
    print('ALL_LOSS_ARMS_COMPLETE',flush=True)

if __name__=='__main__':main()
