import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from engine import read_history,catalog,forecast_materials,simulate

st.set_page_config(page_title='ORIGIN | FTA Risk Intelligence',page_icon='◈',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],div,span,p {font-family:'DM Sans','Noto Sans KR',sans-serif}
.stApp{background:#0b1020;color:#edf2ff}
[data-testid="stSidebar"]{background:#11192c;border-right:1px solid #26314a}
[data-testid="stMetric"]{background:#17233a;padding:18px;border:1px solid #2b3955;border-radius:14px}
[data-testid="stMetricValue"]{color:#f7fafc;font-weight:800}
[data-testid="stMetricLabel"]{color:#a6b5d2}
.hero{padding:26px 28px;border-radius:20px;background:linear-gradient(115deg,#152d49,#14223b 65%,#152d3a);border:1px solid #334967;margin-bottom:20px}
.eyebrow{font-size:11px;letter-spacing:2px;color:#6ee7cc;font-weight:800}
.hero h1{font-size:37px;margin:8px 0 8px;color:#f8fafc;letter-spacing:-1px}
.hero p{color:#bac9e1;font-size:14px;margin:0}
.section{font-size:19px;font-weight:800;color:#f2f6ff;margin:16px 0 10px}
.note{padding:12px 15px;background:#16233a;border-left:3px solid #4bc9b3;color:#b7c8e1;border-radius:7px;font-size:13px}
[data-testid="stDataFrame"]{border:1px solid #2b3955;border-radius:12px}
.stButton>button[kind="primary"]{background:#22b8a3;color:#071821;border:0;font-weight:800;border-radius:10px}
</style>''',unsafe_allow_html=True)

@st.cache_data
def load():
    return read_history(os.path.join(os.path.dirname(__file__),'data','scenario_B.csv'))

@st.cache_data
def get_catalog():
    return catalog(load())

@st.cache_data(show_spinner=False)
def cached_forecasts(ids, months):
    return forecast_materials(load(),list(ids),months)

history=load(); items=get_catalog()
with st.sidebar:
    st.markdown('### ◈ ORIGIN')
    st.caption('FTA RISK INTELLIGENCE  /  DEMO')
    st.divider()
    st.markdown('**01 / 분석 설정**')
    product=st.selectbox('제품 선택', ['직접 구성 (전체 원재료)']+list(items.product_id.unique()))
    months=st.select_slider('미래 예측 기간',options=[1,3,6,12],value=3,format_func=lambda x:f'{x}개월')
    simulations=st.select_slider('몬테카를로 반복',options=[1000,3000,10000],value=3000,format_func=lambda x:f'{x:,}회')
    st.divider()
    st.markdown('**02 / 분석 전제**')
    st.caption('RVC 공제법 · 분석 기준 40% · 원산지 미확인 재료는 보수적으로 VNM 포함')
    st.caption('자료: 시나리오 B, 2023–2025 월별 수익률. 일부 가격은 대체·프록시 데이터일 수 있습니다.')
    st.divider()
    st.caption('데모 목적의 위험 추정이며 FTA 법적 판정 또는 관세 절감 보증이 아닙니다.')

st.markdown('''<div class="hero"><div class="eyebrow">TRADE INTELLIGENCE / ORIGIN COMPLIANCE</div><h1>FTA Origin Risk Prediction</h1><p>RVC 위험 확률 산출 서비스</p></div>''',unsafe_allow_html=True)
left,right=st.columns([1.35,1],gap='large')
with left:
    st.markdown('<div class="section">01. BOM 및 원산지 정보</div>',unsafe_allow_html=True)
    initial=items if product.startswith('직접') else items[items.product_id.eq(product)]
    initial=initial.copy()
    initial['cost_krw']=initial.bom_base_cost_krw.astype(float)
    initial['status']=initial.rvc_treatment.map({'originating':'원산지 확인','non_originating':'비원산지'}).fillna('미확인')
    editable=initial[['material_id','material_name','cost_krw','status']]
    editor=st.data_editor(editable,hide_index=True,use_container_width=True,num_rows='dynamic',key='bom_editor',column_config={
        'material_id':st.column_config.TextColumn('원재료 ID',disabled=True),
        'material_name':st.column_config.TextColumn('원재료명',disabled=True),
        'cost_krw':st.column_config.NumberColumn('제품당 투입비용 (원)',min_value=0,step=10,format='%.2f'),
        'status':st.column_config.SelectboxColumn('원산지 지위',options=['원산지 확인','비원산지','미확인'],required=True),
    })
    st.caption('기준 비용은 제품당 원재료 투입비용입니다. 원산지 지위는 데모 가정이며 실제 증빙을 대체하지 않습니다.')
with right:
    st.markdown('<div class="section">02. FOB 설정</div>',unsafe_allow_html=True)
    default_vnm=float(editor.loc[editor.status.ne('원산지 확인'),'cost_krw'].sum())
    suggested=max(1000,int(np.ceil((default_vnm/.6)/100)*100))
    fob=st.number_input('사용자 설정 FOB (제품당 원)',min_value=1.0,value=float(suggested),step=100.0,format='%.2f')
    st.caption('FOB는 자동 예측하지 않습니다. 직접 입력한 금액을 모든 계산에 적용합니다.')
    vnm=default_vnm
    rvc=(1-vnm/fob)*100
    a,b=st.columns(2)
    a.metric('현재 VNM',f'{vnm:,.0f} 원')
    b.metric('현재 RVC',f'{rvc:,.1f}%',delta=f'{rvc-40:+.1f}%p vs 기준')
    st.progress(float(np.clip(rvc/100,0,1)),text='현재 RVC / 100%')
    if rvc<40: st.error('현재 입력값 기준: RVC 40% 미충족')
    else: st.success('현재 입력값 기준: RVC 40% 이상')
    st.markdown('<div class="note">RVC = (FOB − VNM) ÷ FOB × 100<br>원산지 미확인 재료는 분석상 VNM에 포함됩니다.</div>',unsafe_allow_html=True)

st.divider()
st.markdown('<div class="section">03. 미래 원산지 위험 전망</div>',unsafe_allow_html=True)
run=st.button('◈ ARMA 예측 및 위험 분석 실행',type='primary',use_container_width=True)
if run:
    try:
        editor['cost_krw']=pd.to_numeric(editor.cost_krw,errors='coerce').fillna(0)
        if editor.material_id.isna().any() or editor.material_id.duplicated().any():
            st.error('원재료 ID가 없거나 중복되어 있습니다.'); st.stop()
        if not set(editor.material_id).issubset(set(items.material_id)):
            st.error('현재는 B 데이터에 등록된 원재료만 예측할 수 있습니다.');st.stop()
        ids=tuple(editor.loc[(editor.status.ne('원산지 확인'))&(editor.cost_krw>0),'material_id'].tolist())
        with st.spinner('ARMA 후보 모형 적합 및 몬테카를로 계산 중...'):
            fc=cached_forecasts(ids,months)
            result=simulate(editor,fc,fob,months,simulations)
        st.session_state['result']=result
        st.session_state['result_settings']=(product,months,simulations,float(fob),editor.to_json())
    except Exception as e:
        st.error(f'분석 오류: {e}')
if 'result' in st.session_state:
    result=st.session_state['result']
    saved=st.session_state.get('result_settings')
    current=(product,months,simulations,float(fob),editor.to_json())
    if saved!=current: st.warning('입력값이 변경되었습니다. 최신 결과를 보려면 분석을 다시 실행하세요.')
    if not result['available']:
        st.warning('미래 위험 확률을 계산할 수 없습니다. 데이터가 부족한 재료: '+', '.join(result['missing']))
        st.info('원재료를 원산지 확인으로 변경하거나, 충분한 가격 관측치가 있는 재료로 구성해 다시 실행할 수 있습니다. 실제 원산지 지위는 임의로 변경하면 안 됩니다.')
    else:
        pred=result['monthly']; end=pred.iloc[-1]
        a,b,c=st.columns(3)
        a.metric(f'{months}개월 후 예상 RVC',f'{end.mean_rvc:.1f}%')
        b.metric('RVC 40% 미충족 확률',f'{end.risk:.1f}%')
        c.metric('90% 예측구간',f'{end.lo:.1f}% ~ {end.hi:.1f}%')
        if end.risk>=50: st.error('높은 위험 추정: FOB 또는 비원산지재료의 비용을 재검토하세요.')
        elif end.risk>=30: st.warning('주의: 미충족 위험이 30% 이상으로 추정됩니다.')
        else: st.info('예측 위험이 30% 미만입니다. 위험이 0이라는 뜻은 아닙니다.')
        t1,t2=st.tabs(['RVC 전망','원재료 영향'])
        with t1:
            fig=go.Figure()
            x=pred.month.tolist()
            fig.add_trace(go.Scatter(x=x,y=pred.hi,mode='lines',line=dict(width=0),showlegend=False))
            fig.add_trace(go.Scatter(x=x,y=pred.lo,mode='lines',line=dict(width=0),fill='tonexty',fillcolor='rgba(42,210,178,.18)',name='90% 예측구간'))
            fig.add_trace(go.Scatter(x=x,y=pred.mean_rvc,mode='lines+markers',line=dict(color='#42dec2',width=3),name='예상 RVC'))
            fig.add_hline(y=40,line_dash='dash',line_color='#f6b768',annotation_text='분석 기준 40%')
            fig.update_layout(template='plotly_dark',paper_bgcolor='#0b1020',plot_bgcolor='#101b2e',height=340,margin=dict(l=10,r=10,t=20,b=20),xaxis_title='예측 개월',yaxis_title='RVC (%)')
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(pred.rename(columns={'month':'예측 개월','mean_rvc':'평균 RVC','median_rvc':'중앙 RVC','lo':'5% 분위','hi':'95% 분위','risk':'미충족 확률 (%)','mean_vnm':'평균 VNM'}).round(2),use_container_width=True,hide_index=True)
        with t2:
            if len(result['materials']):
                mat=result['materials'].copy();mat['예상 비용 변화']=mat.forecast_cost-mat.base_cost
                st.dataframe(mat.rename(columns={'material':'원재료','base_cost':'현재 VNM 기여액','forecast_cost':'미래 예상 기여액','model':'선택 ARMA'}).round(2),use_container_width=True,hide_index=True)
            else: st.info('VNM에 포함된 재료가 없습니다.')
        st.download_button('월별 예측 결과 CSV 저장',pred.to_csv(index=False).encode('utf-8-sig'),file_name='rvc_forecast.csv',mime='text/csv')
        st.caption('확률은 과거 잔차의 독립 부트스트랩에 기반한 모의 추정치입니다. 가격 프록시, 짧은 표본, 상관관계 미반영 등으로 실제 위험과 다를 수 있습니다.')
else:
    st.info('위에서 BOM·FOB를 설정한 뒤 「ARMA 예측 및 위험 분석 실행」을 눌러주세요.')
st.divider()
st.caption('© ORIGIN / Prototype · 합성/가정 원산지 지위 및 기준 FOB · 실제 협정별 원산지 결정기준, 누적, 증빙, FOB 산정은 별도 검토 필요')
