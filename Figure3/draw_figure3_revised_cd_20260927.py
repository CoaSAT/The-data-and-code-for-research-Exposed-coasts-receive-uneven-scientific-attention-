"""Publication Figure 3: original a/b data, harmonized site-inventory c/d.

Uses the completed robustness analysis; no fresh statistical choices or changes
to source data. Exports editable vector PDF/SVG, 600-dpi TIFF/PNG and source data.
"""
from pathlib import Path
import sys, os, json
ROOT=Path(__file__).resolve().parent
ROB=ROOT/'data/infrastructure_robustness_20260927'
OUT=ROOT/'data/figure3_revised_cd_20260927'

os.environ['MPLCONFIGDIR']=str(OUT/'_mpl_cache')
import numpy as np
import pandas as pd
import shapely
from scipy.stats import spearmanr,t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon,Rectangle,Patch
from matplotlib.collections import PatchCollection


COLORS={'High-attention reference':'#4F7FA8','Lower-attention comparison':'#BF7B32','Additional comparison':'#4F9F8C'}
NETWORKS={'psmsl':('PSMSL tide gauges','#0072B2','o'),
          'deims':('Coastal DEIMS/ILTER','#009E73','^'),
          'goaon':('GOA-ON platforms','#E69F00','s'),
          'seagrass':('SeagrassNet sites','#CC79A7','D')}



def style(ax,letter,grid=False):
    ax.tick_params(length=2.2,width=.5,pad=2,labelsize=6.7)
    for s in ax.spines.values():s.set_linewidth(.5)
    if grid:ax.grid(color='#E1E7EC',lw=.45);ax.set_axisbelow(True)
    ax.text(-.12,1.02,letter,transform=ax.transAxes,fontsize=13,fontweight='bold',va='bottom',ha='right')


def panel_a(ax,pop):
    ax.plot([0,80],[0,80],color='#ADBBC9',lw=.6)
    specs={'Cold spots':('Coldspots','#E8E0EF',3,5,'left'),
           'Other classes':('Intermediate','#BDB0CB',-3,2,'right'),
           'Hot spots':('Hotspots','#6D488C',3,6,'left')}
    for _,r in pop.iterrows():
        name,c,dx,dy,ha=specs[r.attention_class]
        x,y=r.coastline_share_pct,r.population_share_pct
        ax.scatter(x,y,s=31,color=c,edgecolors='white',lw=.4,zorder=3)
        ax.text(x+dx,y+dy,f'{name}\n{y:.1f}% pop / {x:.1f}% coast',fontsize=6.6,ha=ha,
                bbox={'facecolor':'white','edgecolor':'none','pad':1.5},zorder=4)
    ax.set(xlim=(0,80),ylim=(0,80),xticks=np.arange(0,81,10),yticks=np.arange(0,81,10),
           xlabel='Share of assessed coastline (%)',ylabel='Share of mapped coastal population (%)')
    style(ax,'a',True)


def panel_b(ax,d):
    x=d.log_gdp.to_numpy();y=d.log_attention_intensity.to_numpy()
    X=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(X,y,rcond=None)[0]
    residual=y-X@coef;sse=np.sum(residual**2);r2=1-sse/np.sum((y-y.mean())**2)
    cov=(sse/(len(x)-2))*np.linalg.inv(X.T@X)
    xx=np.linspace(x.min(),x.max(),200);G=np.column_stack([np.ones(len(xx)),xx]);yy=G@coef
    se=np.sqrt(np.einsum('ij,jk,ik->i',G,cov,G));crit=t.ppf(.975,len(x)-2)
    assert len(x)==142 and abs(r2-.152)<.0005
    nz=d.mean_attention_per_gcc_point>0
    ax.fill_between(10**xx,yy-crit*se,yy+crit*se,color='#D9E4EE',lw=0)
    ax.plot(10**xx,yy,color='#306F9B',lw=1.25)
    ax.scatter(10**x[nz],y[nz],s=12,color='#80C7D5',edgecolors='white',lw=.3,zorder=3)
    ax.scatter(10**x[~nz],y[~nz],s=17,facecolors='white',edgecolors='#C87823',lw=.85,zorder=3)
    ax.set(xscale='log',ylim=(-.2,4.2),yticks=np.arange(0,5),xlabel='GDP per capita (PPP, 2017 US$; log scale)',
           ylabel=r'$\log_{10}$(attention intensity + 1)')
    ax.text(.025,.965,f'Ordinary least squares\n$R^2$ = {r2:.3f}; n = {len(d)}',transform=ax.transAxes,
            va='top',fontsize=7.2,fontweight='bold')
    handles=[Line2D([],[],marker='o',mfc='white',mec='#C87823',ms=4,ls='',label=f'Zero recorded attention (n = {(~nz).sum()})'),
             Patch(fc='#D9E4EE',ec='none',label='95% CI of fitted mean')]
    ax.legend(handles=handles,loc='lower right',bbox_to_anchor=(1,.1),fontsize=6.4,frameon=False,handlelength=1.4)
    style(ax,'b',True)


def panel_c(ax,sites,boxes):
    features=json.loads((ROOT/'data/ne_110m_land.geojson').read_text(encoding='utf-8'))['features']
    patches=[]
    for f in features:
        g=f['geometry'];polys=[g['coordinates']] if g['type']=='Polygon' else g['coordinates']
        for p in polys:
            if p:patches.append(Polygon(p[0],closed=True))
    ax.add_collection(PatchCollection(patches,facecolor='#E8EDF1',edgecolor='#C4CDD5',lw=.2,zorder=1))
    for k,(label,color,marker) in NETWORKS.items():
        d=sites[sites.network.eq(k)]
        ax.scatter(d.lon,d.lat,s=2.8 if marker=='o' else 4.0,color=color,marker=marker,
                   edgecolors='white',lw=.12,alpha=.9,zorder=3)
    for _,r in boxes.iterrows():
        ax.add_patch(Rectangle((r.lon_min,r.lat_min),r.lon_max-r.lon_min,r.lat_max-r.lat_min,
                              fc=(1,.85,.35,.06),ec='#222',lw=.65,zorder=4))
        ax.text(r.lon_min+.8,r.lat_max-.5,str(r.no),va='top',fontsize=5.3,
                bbox={'fc':'white','ec':'#444','lw':.3,'pad':.55},zorder=5)
    ax.set(xlim=(-180,180),ylim=(-85,85),xticks=np.arange(-180,181,60),yticks=np.arange(-60,61,30))
    style(ax,'c')
    handles=[Line2D([],[],marker=m,ls='',color=c,ms=3,markeredgecolor='white',markeredgewidth=.2,
              label=f'{label} (n = {sum(sites.network.eq(k)):,})') for k,(label,c,m) in NETWORKS.items()]
    ax.legend(handles=handles,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.105),frameon=False,
              fontsize=5.65,columnspacing=.8,handletextpad=.35,labelspacing=.55)


def panel_d(ax,reg,cum,growth):
    x=reg.index_inventory4.to_numpy();y=reg.log_cumulative_attention.to_numpy()
    # Marker AREA is proportional to change in log counts; no min/max remapping.
    sizes=52*reg.growth_10yr.to_numpy()
    for group,color in COLORS.items():
        k=reg.group.eq(group)
        ax.scatter(x[k],y[k],s=sizes[k],color=color,edgecolors='white',lw=.4,zorder=3)
    coef=np.polyfit(x,y,1);xx=np.linspace(x.min()-.02,x.max()+.02,150)
    ax.plot(xx,np.polyval(coef,xx),color='#176D8F',lw=.9,zorder=2)
    label_xy={
      'Mediterranean Sea':(.64,4.68), 'China coast':(.51,4.55),
      'U.S. East Coast':(.71,4.35),'U.S. West Coast':(.74,4.09),
      'Bay of Bengal':(.29,4.16),'Chile-Peru coast':(.33,3.91),
      'Australia east coast':(.52,3.77),'South-eastern Brazil':(.09,4.00),
      'Gulf of Guinea':(.08,3.65),'East Africa coast':(.08,3.25),
      'Cape of Good Hope':(.31,3.43),'Australia west coast':(.25,3.16)}
    for _,r in reg.iterrows():
        tx,ty=label_xy[r.region]
        ax.text(tx,ty,r.region,ha='left',va='center',fontsize=5.7,zorder=4)
    ax.text(.02,.975,f'Cumulative attention ρ = {cum.rho:.2f} (95% CI {cum.ci_low:.2f}–{cum.ci_high:.2f})\n'
            f'Change in attention ρ = {growth.rho:.2f} (95% CI {growth.ci_low:.2f} to {growth.ci_high:.2f})',
            transform=ax.transAxes,va='top',fontsize=6.35,fontweight='bold',linespacing=1.6)
    ax.set(xlim=(.065,1.005),ylim=(3.06,5.05),xticks=np.arange(.1,1,.1),yticks=np.arange(3.2,4.8,.2),
            xlabel='Observation-site inventory index',ylabel='Cumulative attention through 2025\n'+r'$\log_{10}$(unique records + 1)')
    style(ax,'d')
    # Put keys in the unoccupied lower-right part of the panel.
    handles=[Line2D([],[],marker='o',mfc=c,mec='white',mew=.3,ms=3.5,ls='',label=label)
      for c,label in zip(COLORS.values(),['High-attention reference regions','Lower-attention comparison regions','Additional comparison regions'])]
    leg=ax.legend(handles=handles,loc='lower right',bbox_to_anchor=(1,.015),fontsize=5.4,
                  frameon=False,handletextpad=.35,borderaxespad=.25,labelspacing=.35)
    ax.add_artist(leg)
    size_handles=[Line2D([],[],marker='o',ms=np.sqrt(52*v),mfc='none',mec='#444',mew=.4,ls='',label=f'{v:.1f}') for v in [.2,.5,.8]]
    ax.legend(handles=size_handles,loc='upper center',bbox_to_anchor=(.5,-.245),ncol=3,fontsize=5.7,
              title=r'$\Delta\log_{10}(N + 1)$'+': 2016–2025 versus 2000–2009',title_fontsize=5.6,
              frameon=False,columnspacing=.65,handletextpad=.3,borderaxespad=.2,labelspacing=.45)


def export(fig,name):
    fig.savefig(OUT/(name+'.pdf'),facecolor='white')
    fig.savefig(OUT/(name+'.svg'),facecolor='white')
    fig.savefig(OUT/(name+'.png'),dpi=300,facecolor='white')
    fig.savefig(OUT/(name+'_600dpi.png'),dpi=600,facecolor='white')
    fig.savefig(OUT/(name+'_600dpi.tif'),dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})




