#!/usr/bin/env python3
"""Static scientific coverage figures from the reviewed development-only audit."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preview',type=Path,required=True);a=p.parse_args()
    raw=a.report.read_bytes();r=json.loads(raw)
    assert r['status']=='DEVELOPMENT_DIAGNOSTICS_NOT_PROTOCOL_FREEZE' and r['all_candidate_cases']==159084
    a.output.mkdir(parents=True,exist_ok=True);a.preview.mkdir(parents=True,exist_ok=True)
    colors=('#bf5700','#386f87');names={'03161000':'Jefferson','03164000':'Galax','03479000':'Sugar Grove'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,4.7),sharey=True)
    for ax,(gauge,name) in zip(axes,names.items()):
        rows={x['season_utc']:x for x in r['seasonal_coverage'] if x['gauge_id']==gauge and x['split']=='candidate_validation'}
        seasons=['DJF','MAM','JJA','SON'];x=np.arange(4)
        common=[100*rows[s]['common_L168_fraction'] for s in seasons];natural=[100*rows[s]['natural_L24_fraction'] for s in seasons]
        for delta,values,color,label,hatch in ((-.18,common,colors[0],'Common L168',''),(.18,natural,colors[1],'Natural L24','//')):
            bars=ax.bar(x+delta,values,.34,color=color,label=label,hatch=hatch,edgecolor='white',linewidth=.5)
            ax.bar_label(bars,labels=[f'{v:.0f}%' if abs(common[i]-natural[i])>.05 else '' for i,v in enumerate(values)],fontsize=8,padding=3)
        for i,(c,n) in enumerate(zip(common,natural)):
            if abs(c-n)<=.05:ax.text(i,max(c,n)+2.8,f'{c:.0f}% both',ha='center',fontsize=8)
        ax.set_xticks(x,seasons);ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_title(name+'\n'+gauge,fontsize=11)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Complete-case coverage (%)')
    fig.suptitle('Development-validation case availability by season',fontsize=16,y=.98)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.86),ncol=2,frameon=False)
    fig.subplots_adjust(top=.68,bottom=.23,wspace=.18)
    fig.text(.04,.10,'2023-07-31 to 2024-07-31 UTC; origin-season denominators after H18 boundary purging.\nBoth cohorts require complete weather, soil and H18 reference labels. These are availability counts, not forecast scores.',fontsize=9)
    for suffix,folder in (('svg',a.output),('png',a.preview)):
        path=folder/('seasonal-coverage.'+suffix);assert not path.exists();fig.savefig(path,dpi=160,facecolor='white')
    plt.close(fig)
    rows={x['gauge_id']:next(y for y in x['target_cell_regime_coverage'] if y['regime']=='HIGH')
          for x in r['fold_regime_and_event_diagnostics'] if x['fold']=='final_development'}
    fig,ax=plt.subplots(figsize=(8.6,5.1));x=np.arange(3)
    for delta,key,color,label,hatch in ((-.19,'common_comparison_cells',colors[0],'Common L168',''),(.19,'natural_L24_comparison_cells',colors[1],'Natural L24','//')):
        values=[100*rows[g][key]/rows[g]['reference_supported_cells'] for g in names]
        bars=ax.bar(x+delta,values,.36,color=color,label=label,hatch=hatch,edgecolor='white')
        ax.bar_label(bars,labels=[f'{v:.1f}%' for v in values],padding=4,fontsize=10)
    ax.set_xticks(x,list(names.values()));ax.set_ylim(0,114);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Retained high-flow target cells (%)')
    ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.14),ncol=2,frameon=False)
    fig.suptitle('High observed-flow coverage in development validation',fontsize=14,y=.98)
    fig.text(.5,.92,'2023-07-31 to 2024-07-31 UTC',ha='center',fontsize=11)
    fig.subplots_adjust(top=.77,bottom=.24,left=.11,right=.98)
    fig.text(.06,.11,'High flow: approved Q above gradient-training q95. Denominators: 3,456 / 3,474 / 6,048 cells.\nCells repeat reference observations across forecast origins/leads; they are not independent events.\nMatched comparisons retain common L168 support. Natural L24 adds coverage, not a skill finding.',fontsize=9)
    for suffix,folder in (('svg',a.output),('png',a.preview)):
        path=folder/('high-flow-coverage.'+suffix);assert not path.exists();fig.savefig(path,dpi=160,facecolor='white')
    plt.close(fig)
    receipt={'schema':'hydra-coverage-figures/v1','source_report_sha256':hashlib.sha256(raw).hexdigest(),
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,
             'figures':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.output.glob('*.svg'))},
             'scope':'DEVELOPMENT_AVAILABILITY_NOT_FORECAST_SKILL','protected_data':False,'model_predictions_used':False}
    path=a.output/'sources.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
