"""User-requested wide 600 dpi Figure 3, without region key or panel-d legends."""
import draw_figure3_equal_windows as source
from matplotlib.text import Text
from PIL import Image

base=source.base
plt=source.plt
pd=source.pd
OUT=base.ROOT/'outputs'
OUT.mkdir(exist_ok=True)

def main():
    data=source.OUT
    reg=pd.read_csv(data/'Figure3d_1000km_source_data.csv')
    sites=pd.read_csv(data/'Figure3c_network_sites.csv')
    pop=pd.read_csv(data/'Figure3a_source_data.csv')
    gdp=pd.read_csv(data/'Figure3b_source_data.csv')
    features=source.json.loads((data/'Figure3c_1000km_footprints.geojson').read_text())['features']
    stats=pd.read_csv(base.ROB/'correlation_summary.csv').set_index('analysis')
    cum=stats.loc['radius_1000km__inventory4__cumulative']
    growth=stats.loc['radius_1000km__inventory4__growth_10yr']
    plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7.1,
                        'axes.labelpad':3,'pdf.fonttype':42,'svg.fonttype':'none'})
    base.OUT=OUT
    for full in (True,False):
        fig=plt.figure(figsize=(240/25.4,(128 if full else 66)/25.4))
        if full:
            a=fig.add_axes([.075,.575,.36,.36])
            b=fig.add_axes([.51,.575,.475,.36])
            base.panel_a(a,pop);base.panel_b(b,gdp)
            a.set_ylabel('Share of mapped coastal\npopulation (%)')
            b.set_xlabel('GDP per capita (PPP, constant 2017 international dollar)')
            bottom=.135;height=.345
        else:bottom=.23;height=.67
        c=fig.add_axes([.075,bottom,.36,height])
        d=fig.add_axes([.51,bottom,.475,height])
        source.panel_c(c,sites,reg,features)
        source.panel_d(d,reg,cum,growth)
        d.set_ylabel('Cumulative attention\n'+r'$\log_{10}$(unique records + 1)')
        for txt in d.texts:
            if txt.get_text().startswith('Cumulative attention ρ'):
                txt.set_text(txt.get_text().replace('Cumulative attention ρ','Cumulative attention (through 2025) ρ'))
        # All statistical inputs and encodings are retained; only layout changes.
        for txt in fig.findobj(match=Text):txt.set_fontsize(txt.get_fontsize()*1.45)
        name='Figure3_1000km_wide' if full else 'Figure3cd_1000km_wide'
        base.export(fig,name)
        plt.close(fig)
        for suffix in ('_600dpi.png','_600dpi.tif'):
            with Image.open(OUT/(name+suffix)) as im:
                dpi=im.info.get('dpi')
                assert dpi and all(abs(v-600)<.1 for v in dpi)
                print(name+suffix,im.size,dpi)

if __name__=='__main__':main()
