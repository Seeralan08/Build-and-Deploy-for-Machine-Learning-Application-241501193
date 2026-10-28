"""Local interactive demo for the synthetic India EV charging demand study.
Run `streamlit run app.py` after installing requirements. No internet or API key needed once installed.
"""
from pathlib import Path
import csv
import numpy as np
import streamlit as st
BASE=Path(__file__).parent
rows=list(csv.DictReader((BASE/'data'/'india_ev_charging_demand_synthetic.csv').open(encoding='utf-8')))
FEATURES=['hour','is_weekend','temperature_c','rainfall_mm','charger_ports','event_evening','latitude','longitude']
X=np.array([[float(r[k]) for k in FEATURES] for r in rows]); y=np.array([float(r['sessions']) for r in rows]); n=int(.8*len(y)); mean=X[:n].mean(0); scale=X[:n].std(0); scale[scale==0]=1
Z=np.column_stack([np.ones(len(y)),(X-mean)/scale]); coef=np.linalg.lstsq(Z[:n],y[:n],rcond=None)[0]
st.set_page_config(page_title='ChargePulse | India',page_icon='🔌',layout='wide')
st.markdown("""<style>.stApp{background:#f3f7f6}.block-container{padding-top:2rem}.hero{background:linear-gradient(115deg,#073b3a,#0b6e69);padding:26px 30px;border-radius:18px;color:#fff;margin-bottom:18px}.hero h1{color:white;margin:0}.hero p{color:#d8efea;margin:7px 0 0}</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>ChargePulse | India</h1><p>Offline EV charging demand estimator · BDML mini project · illustrative synthetic dataset</p></div>',unsafe_allow_html=True)
left,right=st.columns([1,1.5],gap='large')
with left:
 st.subheader('Scenario inputs')
 city=st.selectbox('Station cluster',[r['station_cluster'] for r in rows[:24]])
 sample=next(r for r in rows if r['station_cluster']==city)
 hour=st.slider('Hour of day',0,23,18); weekend=st.toggle('Weekend',value=False); temp=st.slider('Temperature (°C)',10.0,45.0,29.0); rain=st.slider('Rainfall (mm/hour)',0.0,30.0,0.0); event=st.toggle('Large evening event',value=False)
with right:
 st.subheader('Estimated demand')
 row=[hour,int(weekend),temp,rain,float(sample['charger_ports']),int(event),float(sample['latitude']),float(sample['longitude'])]
 z=np.r_[1,(np.array(row)-mean)/scale]; estimate=max(0,float(z@coef))
 st.metric('Expected charging sessions / station-hour',f'{estimate:.1f}')
 st.caption(f"Cluster: {city} · {sample['charger_ports']} ports. Estimate is a model output on simulated teaching data, not an operational prediction.")
 st.image(str(BASE/'assets'/'hourly_profile.png'),use_container_width=True)
st.divider(); st.markdown('**Model note:** NumPy least-squares regression with an 80/20 chronological split. Dataset is synthetic and generated deterministically; this local demo does not call a network service.')
