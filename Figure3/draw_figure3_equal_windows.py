"""Render Figure 3 with matched equal-radius map footprints and statistics."""
import draw_figure3_revised_cd_20260927 as base
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shapely
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

OUT=base.ROOT/'data/figure3_equal_windows_20260927'
OUT.mkdir(exist_ok=True)

def panel_c(ax,sites,reg,features):
    base.panel_c(ax,sites,pd.DataFrame())
    for f in features:
        r=reg.set_index('region').loc[f['properties']['region']]
        ax.add_patch(Polygon(f['geometry']['coordinates'][0],fc=(1,.85,.35,.1),ec='#333333',lw=.7,zorder=4))
        ax.annotate(str(int(r.no)),(r.anchor_lon,r.anchor_lat),xytext=(3,3),textcoords='offset points',
                    fontsize=6,fontweight='bold',bbox=dict(fc='white',ec='#555',lw=.3,pad=.8),zorder=5)

def panel_d(ax,reg,cum,growth):
    x=reg.index_inventory4.to_numpy();y=np.log10(reg.cumulative_through_2025+1).to_numpy()
    sizes=65*reg.growth_10yr.to_numpy()
    assert (sizes>0).all()
    for group,color in base.COLORS.items():
        mask=reg.group.eq(group)
        ax.scatter(x[mask],y[mask],s=sizes[mask],c=color,edgecolors='white',lw=.5,zorder=3)
    # Numbers share a lookup with the map; point labels no longer cover the fit.
    offsets={1:(-5,8),2:(8,0),3:(6,-8),4:(7,0),5:(-9,5),6:(-8,7),7:(-9,-6),8:(7,-3),9:(-7,7),10:(-8,-9),11:(7,2),12:(8,-8)}
    for i,(_,r) in enumerate(reg.iterrows()):
        ax.annotate(str(int(r.no)),(x[i],y[i]),xytext=offsets[int(r.no)],textcoords='offset points',
                    fontsize=6.1,fontweight='bold',ha='center',va='center',zorder=4,
                    bbox=dict(fc='white',ec='none',pad=.4))
    xx=np.linspace(max(0,x.min()-.025),x.max()+.025,100)
    ax.plot(xx,np.polyval(np.polyfit(x,y,1),xx),c='#176D8F',lw=.9,zorder=2)
    ax.set(xlim=(0,.95),ylim=(2.65,5.05),xticks=np.arange(0,1,.2),yticks=np.arange(2.8,4.5,.4),
           xlabel='Observation-site inventory index',ylabel='Cumulative attention through 2025\n'+r'$\log_{10}$(unique records + 1)')
    ax.text(.025,.97,f'Cumulative attention ρ = {cum.rho:.2f} (95% CI {cum.ci_low:.2f}–{cum.ci_high:.2f})\n'
            f'Change in attention ρ = {growth.rho:.2f} (95% CI {growth.ci_low:.2f} to {growth.ci_high:.2f})',
            transform=ax.transAxes,va='top',fontsize=6.2,linespacing=1.6,fontweight='bold')
    base.style(ax,'d')

def footer(fig,reg,map_ax,scatter_ax):
    # Dedicated physical bands for size and colour keys, independent of data limits.
    p=scatter_ax.get_position();h=fig.get_figheight()*72
    colorax=fig.add_axes([p.x0,p.y0-53/h,p.width,22/h]);colorax.axis('off')
    handles=[Line2D([],[],marker='o',color=c,ls='',ms=3.5,label=l) for c,l in zip(base.COLORS.values(),
        ['High-attention reference regions','Lower-attention comparison regions','Additional comparison regions'])]
    colorax.legend(handles=handles,loc='center',frameon=False,fontsize=5.6,labelspacing=.4,handletextpad=.4)
    sizeax=fig.add_axes([p.x0,p.y0-82/h,p.width,26/h]);sizeax.axis('off')
    handles=[Line2D([],[],marker='o',ms=np.sqrt(65*v),mfc='none',mec='#444',mew=.4,ls='',label=f'{v:.1f}') for v in [.2,.5,.8]]
    sizeax.legend(handles=handles,loc='center',ncol=3,frameon=False,fontsize=5.8,
        title=r'$\Delta\log_{10}(N+1)$: 2016–2025 versus 2000–2009',title_fontsize=5.8)
    # Name key sits below the map, away from both map and scatter axes.
    p=map_ax.get_position()
    for j,(_,r) in enumerate(reg.sort_values('no').iterrows()):
        col=j//6;row=j%6
        fig.text(p.x0+col*.19,p.y0-(47+9*row)/h,f'{int(r.no)}  {r.region}',fontsize=5.7,va='top')


