"""Rainbow distributions with identical frames and colors across methods.

Extends the existing ear rainbow plots; no coordinate averaging for network panels.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
TAG='zeroshot-cheese-kondo-c3d-kauf-vits-seeds012'
SESSION='9b528ad0-4599-4a55-9148-96cc1d93fb24'
PARTS=['base','top','tip','bottom']
COLORS=['#00d9ff','#ff9900','#ff3cdf','#6aff46']
METHODS=['Single network: seed 0','Single network: seed 1','Single network: seed 2','Ensemble median','Pose-EKS (rank 4)']
ROIS={'left':(168,0,288,96),'right':(78,22,198,118)}

def main():
 cv2.setNumThreads(1);data={};stats=[];audit={}
 for camera in ['left','right']:
  arrays=[];conf=[];keys=[f'ear_{p}_{camera}' for p in PARTS]
  for seed in range(3):
   d=pd.read_csv(next((ROOT.parent/'ibl-ear-rainbow-9b528ad0-10-01/csv/predictions').glob(f'*_{camera}_seed{seed}.csv')),header=[0,1,2],index_col=0).droplevel(0,axis=1)
   assert np.array_equal(d.index,np.arange(5000))
   arrays.append(np.stack([d[k][['x','y']].values for k in keys],axis=1))
   conf.append(np.stack([d[k]['likelihood'].values for k in keys],axis=1))
  raw=np.stack(arrays);confidence=np.stack(conf)
  model=np.load(ROOT/f'{TAG}_{camera}_latent-model.npz')
  median=np.median(raw,axis=0);np.testing.assert_allclose(median.reshape(5000,8),model['observation_medians'])
  eks=(model['mean']+model['smoothed_means']@model['W'].T).reshape(5000,4,2)
  xy=np.concatenate([raw,median[None],eks[None]],axis=0)
  source=Path('/teamspace/studios/this_studio/poseinterface/data/head-fixed-v9/videos')/f'_iblrig_{camera}Camera.downsampled.{SESSION}.mp4'
  cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,2500);ok,img=cap.read();cap.release();assert ok
  data[camera]=(xy,confidence,cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
  audit[camera]=dict(background_source=str(source),background_frame=2500,keypoints=keys)
 for gated in [False,True]:
  mode='matched-conf-gt07' if gated else 'ungated'
  fig,axes=plt.subplots(2,5,figsize=(25,12),facecolor='#141414')
  for row,camera in enumerate(['left','right']):
   xy,confidence,img=data[camera]
   keep=(confidence>.7).all(axis=0) if gated else np.ones((5000,4),bool)
   counts=keep.sum(0)
   audit[camera][mode]=dict(retained_per_keypoint=dict(zip(PARTS,counts.tolist())),candidates=5000)
   x0,y0,x1,y1=ROIS[camera]
   for m,method in enumerate(METHODS):
    ax=axes[row,m];ax.imshow(img)
    pts=xy[m].reshape(-1,2)[keep.ravel()];colors=np.tile(COLORS,5000)[keep.ravel()]
    order=np.random.default_rng(19).permutation(len(pts))
    ax.scatter(pts[order,0],pts[order,1],c=colors[order],s=1.4,alpha=.18,linewidths=0)
    ax.set_xlim(-.5,319.5);ax.set_ylim(255.5,-.5);ax.axis('off')
    ax.set_title(f'{camera.capitalize()} | {method}',color='white',fontsize=13)
    for k,(part,color) in enumerate(zip(PARTS,COLORS)):
     ax.text(.98,.97-.05*k,f'{part}: {counts[k]:,}/5,000',transform=ax.transAxes,ha='right',va='top',color=color,fontsize=8,bbox=dict(facecolor='black',alpha=.65,edgecolor='none',pad=1))
     z=xy[m,:,k][keep[:,k]]
     if len(z):
      center=np.median(z,axis=0);radius=np.linalg.norm(z-center,axis=1)
      outside=(z[:,0]<x0)|(z[:,0]>=x1)|(z[:,1]<y0)|(z[:,1]>=y1)
      stats.append(dict(camera=camera,mode=mode,method=method,keypoint=part,n=len(z),outside_broad_ear_crop=int(outside.sum()),outside_fraction=float(outside.mean()),median_x=float(center[0]),median_y=float(center[1]),radius_p95_px=float(np.quantile(radius,.95))))
    ax.plot([12,32],[243,243],color='white',lw=2);ax.text(12,237,'20 px',color='white',fontsize=8)
  gate='All 5,000 frames; no confidence filtering' if not gated else 'Matched frames per landmark: ALL THREE member likelihoods >0.7; same mask for every method'
  fig.suptitle(f'Session 9b528ad0 | ear distributions, frames 0–4999\n{gate}',color='white',fontsize=18,y=.98)
  handles=[Line2D([],[],marker='o',linestyle='none',color=c,label=p.capitalize()) for p,c in zip(PARTS,COLORS)]
  fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.049),ncol=4,labelcolor='white',facecolor='#222222',fontsize=13)
  fig.text(.5,.015,'Zero-shot cheese+kondo+c3d+kauf ViT-S | 5,000 predictions per landmark per panel | native views | background frame 2500\nSame colors, marker size and transparency | pose-EKS fit independently per camera using mixture variance | no ground truth',ha='center',color='white',fontsize=11)
  fig.subplots_adjust(top=.9,bottom=.14,left=.015,right=.985,hspace=.17,wspace=.055)
  f=ROOT/'figures'/f'ear-rainbow-single-networks-vs-pose-eks-{mode}__{TAG}_frames000000-004999.png'
  fig.savefig(f,dpi=160,facecolor=fig.get_facecolor());plt.close(fig);print(f,flush=True)
 pd.DataFrame(stats).to_csv(ROOT/'csv/distribution_diagnostics.csv',index=False)
 (ROOT/'figure-provenance.json').write_text(json.dumps(dict(cameras=audit,point_size=1.4,alpha=.18,broad_ear_crops=ROIS,roi_caveat='coarse image-region diagnostic, not anatomical labels or accuracy'),indent=2)+'\n')
if __name__=='__main__':main()
