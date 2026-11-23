"""Full Figure 2 using original b/c data and highlighted panel a."""
import draw_figure2a_high_exposure_focus as panel_a
import numpy as np
from matplotlib.colors import LinearSegmentedColormap,Normalize
from matplotlib.cm import ScalarMappable
from draw_categorical_blindspot_burden_map import draw_land
from draw_figure2_ipcc_typology import draw_ipcc_map
plt=panel_a.plt;pd=panel_a.pd;ROOT=panel_a.ROOT
OUT=ROOT/'outputs'
OUT.mkdir(exist_ok=True)

def main():
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':9,
                        'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(240/25.4,155/25.4))
    panel_a.main(fig,[.015,.505,.97,.49],export=False)
    df=pd.read_csv(ROOT/'data/figure3a_theme_bbi_20260914/figure3a_theme_bbi_points.csv',usecols=['lon','lat','BBI'])
    assert df.BBI.notna().sum()==655275
    ipcc=pd.read_csv(ROOT/'data/figure2_new_data/figure2_new_data_ipcc_summary.csv')
    ax=fig.add_axes([.065,.165,.495,.295]);draw_land(ax)
    cmap=LinearSegmentedColormap.from_list('burden',['#2A6FBB','#50B4A4','#F2D16B','#E47B36','#8B1E3F'])
    missing=df.BBI.isna()
    ax.scatter(df.loc[missing,'lon'],df.loc[missing,'lat'],s=.18,c='#909AA3',lw=0,rasterized=True)
    valid=df.loc[~missing].sort_values('BBI')
    ax.scatter(valid.lon,valid.lat,s=.3,c=valid.BBI,cmap=cmap,norm=Normalize(0,2),lw=0,rasterized=True)
    ax.set(xlim=(-180,180),ylim=(-65,82),xticks=np.arange(-180,181,60),yticks=np.arange(-60,81,30))
    ax.grid(color='#E7ECEF',lw=.35);ax.set_axisbelow(True)
    for sp in ax.spines.values():sp.set_linewidth(.5);sp.set_color('#929BA2')
    ax.tick_params(labelsize=8,length=2,width=.5)
    fig.text(.012,.468,'b',fontsize=17,fontweight='bold')
    cax=fig.add_axes([.105,.075,.41,.016]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,2),cmap=cmap),cax=cax,orientation='horizontal')
    cb.set_ticks([0,.5,1,1.5,2]);cb.ax.tick_params(labelsize=7.5,length=2)
    cb.outline.set_linewidth(.4)
    cax.set_title('Blind-spot burden per GCC point',fontsize=9,pad=7)
    fig.text(.105,.032,'Evidence-aligned',fontsize=8,fontweight='bold',ha='center')
    fig.text(.515,.032,'Priority evidence gap',fontsize=8,fontweight='bold',ha='center')
    bx=fig.add_axes([.72,.075,.255,.405])
    top=ipcc.sort_values('paper_count',ascending=False).head(12).iloc[::-1]
    vals=top.paper_count.to_numpy();assert list(vals[-2:])==[74133,76393]
    cmap2=LinearSegmentedColormap.from_list('attention',['#CBE4E4','#79BBBD','#226584'])
    bx.barh(top.ipcc_ar6_Name,vals,color=cmap2((vals-vals.min())/(vals.max()-vals.min())),height=.68)
    for y,v in enumerate(vals):bx.text(v+900,y,f'{v:,}',va='center',fontsize=7.2,fontweight='bold')
    bx.set(xlim=(0,88000),xticks=[0,20000,40000,60000,80000],xlabel='Located literature records')
    bx.xaxis.label.set_size(9);bx.tick_params(axis='y',length=0,labelsize=8);bx.tick_params(axis='x',labelsize=7.5,length=2)
    bx.grid(axis='x',color='#E7ECEF',lw=.4);bx.set_axisbelow(True)
    for sp in ['top','right','left']:bx.spines[sp].set_visible(False)
    bx.spines['bottom'].set_color('#AEB6BE');bx.spines['bottom'].set_linewidth(.5)
    fig.text(.593,.468,'c',fontsize=17,fontweight='bold')
    inset=fig.add_axes([.838,.105,.137,.102]);m=draw_ipcc_map(inset,ipcc);m.set_cmap(cmap2)
    inset.set(xlim=(-180,180),ylim=(-65,85),xticks=[],yticks=[]);inset.set_aspect('auto')
    for sp in inset.spines.values():sp.set_linewidth(.5)
    for dpi in [200,600]:fig.savefig(OUT/f'Figure2_revised_{dpi}dpi.png',dpi=dpi,facecolor='white')
    fig.savefig(OUT/'Figure2_revised.pdf',dpi=600,facecolor='white')
    fig.savefig(OUT/'Figure2_revised_600dpi.tif',dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    ipcc.to_csv(OUT/'panel_c_source_data.csv',index=False)
    print(OUT)

if __name__=='__main__':main()
