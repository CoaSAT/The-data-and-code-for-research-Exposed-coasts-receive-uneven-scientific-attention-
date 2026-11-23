from pathlib import Path
import draw_figure1 as p
ROOT=Path(__file__).resolve().parent
(ROOT/'outputs').mkdir(exist_ok=True)
p.configure_attention_bins(150,[600,1500],3000)
p.draw_figure(ROOT/'data/gcc_attention_2deg/gcc_attention_from_2deg_buffer_window.csv',None,ROOT/'outputs/Figure1.png',ROOT/'data/ne_110m_land.geojson',ROOT/'data/figure1_caci/figure1_caci_lorenz_curve_downsampled.csv')
