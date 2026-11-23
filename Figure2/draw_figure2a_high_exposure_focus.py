"""Restyle original five-column flows; emphasize conditional low-attention shares."""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parent

OUT=ROOT/'data/figure2a_high_exposure_focus_20260927'
OUT.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'_mpl_cache')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,PathPatch
from matplotlib.path import Path as MPath

SRC=ROOT/'data/figure2_relayout_sankey_capacity_infra/figure2a_exposure_theme_global_attention_sankey_flows.csv'
ORDER=['Low','Moderate','High'];GLOBAL=['Cold spots','Other classes','Hot spots']
ORANGE='#C87832'

def main(fig=None, rect=None, export=True):
    f=pd.read_csv(SRC)
    def table(domain,stage):
        return f[f.domain.eq(domain)&f.stage.eq(stage)].pivot(index='source',columns='target',values='gcc_points')
    e=table('ESL','exposure_to_theme_attention').loc[ORDER,ORDER]
    eg=table('ESL','theme_attention_to_global_attention').loc[ORDER,GLOBAL]
    u=table('Eutrophication','exposure_to_theme_attention').loc[ORDER,ORDER]
    ug=table('Eutrophication','theme_attention_to_global_attention').loc[ORDER,GLOBAL]
    matrices=[e,eg,ug.T,u.T]
    total=int(e.to_numpy().sum());assert total==655275
    for mat in matrices:assert mat.to_numpy().sum()==total
    for left,right in zip(matrices[:-1],matrices[1:]):assert (left.sum(axis=0)==right.sum(axis=1)).all()
    percentages=[100*e.loc['High','Low']/e.loc['High'].sum(),100*u.loc['High','Low']/u.loc['High'].sum()]
    assert [round(v,1) for v in percentages]==[51.4,38.6]
    totals=[e.sum(axis=1),e.sum(axis=0),eg.sum(axis=0),u.sum(axis=0),u.sum(axis=1)]
    colors=[['#DCE6EC','#8DAFC2','#356480'],['#DDEBE8','#8CB6AF','#387F77'],
            ['#E9E3EC','#BEB0C7','#80658F'],['#DDEBE8','#8CB6AF','#387F77'],['#DCE6EC','#8DAFC2','#356480']]
    xs=[.035,.245,.49,.72,.935];w=.023;scale=.53/total;positions=[]
    for tt in totals:
        pos={};top=.845
        for k,v in tt.items():
            pos[k]=(top-v*scale,top);top-=v*scale+.024
        positions.append(pos)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    if fig is None:fig=plt.figure(figsize=(240/25.4,88/25.4))
    ax=fig.add_axes(rect or [.015,.015,.97,.97]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    highlights=[]
    def ribbon(x0,a,b,x1,c,d,color,alpha,z):
        dx=x1-x0
        verts=[(x0,a),(x0+.48*dx,a),(x0+.52*dx,c),(x1,c),(x1,d),(x0+.52*dx,d),(x0+.48*dx,b),(x0,b),(x0,a)]
        codes=[1,4,4,4,2,4,4,4,79]
        ax.add_patch(PathPatch(MPath(verts,codes),facecolor=color,edgecolor='none',alpha=alpha,zorder=z))
    for j,mat in enumerate(matrices):
        lc={k:v[1] for k,v in positions[j].items()};rc={k:v[1] for k,v in positions[j+1].items()}
        for s in mat.index:
            for t in mat.columns:
                h=mat.loc[s,t]*scale
                args=(xs[j]+w,lc[s]-h,lc[s],xs[j+1],rc[t]-h,rc[t])
                focus=(j==0 and s=='High' and t=='Low') or (j==3 and s=='Low' and t=='High')
                if focus:highlights.append(args)
                else:ribbon(*args,'#B9C9D0' if j in (0,3) else '#CCD1D9',.32 if j in (0,3) else .19,1)
                lc[s]-=h;rc[t]-=h
    for args in highlights:ribbon(*args,ORANGE,.87,2)
    for j,(tt,pos) in enumerate(zip(totals,positions)):
        for i,(k,v) in enumerate(tt.items()):
            a,b=pos[k];ax.add_patch(Rectangle((xs[j],a),w,b-a,fc=colors[j][i],ec='white',lw=.6,zorder=4))
            label={'Cold spots':'Cold','Other classes':'Other','Hot spots':'Hot'}.get(k,k)
            ha='right' if j>=3 else 'left';tx=xs[j]-.006 if j>=3 else xs[j]+w+.006
            ax.text(tx,(a+b)/2,f'{label} {v/total*100:.1f}%',ha=ha,va='center',fontsize=8.1,
                    bbox=dict(fc='white',ec='none',alpha=.82,pad=.5),zorder=5)
    headers=[('Extreme sea-level\nexposure',.065,'left'),('Extreme sea-level\nresearch attention',.27,'center'),
             ('Overall scientific\nattention',.5,'center'),('Eutrophication\nresearch attention',.735,'center'),('Eutrophication\nexposure',.96,'right')]
    for text,x,ha in headers:ax.text(x,.965,text,ha=ha,va='top',fontsize=9,fontweight='bold')
    ax.text(.005,.985,'a',fontsize=17,fontweight='bold',va='top')
    # Keep the highlighted ribbons but remove the callouts and their footer band.
    ax.set_ylim(.24,1)
    if not export:return ax
    for ext in ['pdf','svg']:fig.savefig(OUT/f'Figure2a_high_exposure_focus.{ext}',facecolor='white')
    for dpi in [200,600]:fig.savefig(OUT/f'Figure2a_high_exposure_focus_{dpi}dpi.png',dpi=dpi,facecolor='white')
    fig.savefig(OUT/'Figure2a_high_exposure_focus_600dpi.tif',dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    f.to_csv(OUT/'source_flows.csv',index=False)
    audit={'source':str(SRC),'total':total,'esl_high_low':int(e.loc['High','Low']),'esl_high_total':int(e.loc['High'].sum()),
           'eut_high_low':int(u.loc['High','Low']),'eut_high_total':int(u.loc['High'].sum()),'conditional_percentages':percentages,
           'validation':'Every stage sums to the common denominator; all adjacent node marginals match; highlighted proportions reproduce manuscript rounding.'}
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
