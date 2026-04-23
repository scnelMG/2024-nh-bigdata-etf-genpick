import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd



# 데이터 불러오기
ETF_SOR_IFO = pd.read_csv('./data/본선/NH_CONTEST_ETF_SOR_IFO.csv', encoding='cp949')
HISTORICAL_DIVIDEND = pd.read_csv('./data/본선/NH_CONTEST_DATA_HISTORICAL_DIVIDEND.csv', encoding='cp949')
CUS_TP_IFO = pd.read_csv('./data/본선/NH_CONTEST_NHDATA_CUS_TP_IFO.csv', encoding='cp949')
ETF_HOLDINGS = pd.read_csv('./data/본선/NH_CONTEST_DATA_ETF_HOLDINGS.csv', encoding='cp949')

# IFW_OFW_IFO = pd.read_csv('./data/본선/NH_CONTEST_NHDATA_IFW_OFW_IFO.csv', encoding='cp949')
# STK_DD_IFO = pd.read_csv('./data/본선/NH_CONTEST_NHDATA_STK_DD_IFO.csv', encoding='cp949')
# NW_FC_STK_IEM_IFO = pd.read_csv('./data/본선/NH_CONTEST_NW_FC_STK_IEM_IFO.csv', encoding='cp949')
# STK_DT_QUT = pd.read_csv('./data/본선/NH_CONTEST_STK_DT_QUT.csv', encoding='cp949')


# In[3]:


# 종목 코드 뒤 공백 제거
ETF_SOR_IFO['etf_iem_cd'] = ETF_SOR_IFO['etf_iem_cd'].str.rstrip()
CUS_TP_IFO['tck_iem_cd'] = CUS_TP_IFO['tck_iem_cd'].str.rstrip()
# IFW_OFW_IFO['tck_iem_cd'] = IFW_OFW_IFO['tck_iem_cd'].str.rstrip()
# STK_DD_IFO['tck_iem_cd'] = STK_DD_IFO['tck_iem_cd'].str.rstrip()
# STK_DT_QUT['tck_iem_cd'] = STK_DT_QUT['tck_iem_cd'].str.rstrip()


# In[4]:


# 사용할 날짜 정의
# 날짜가 표기된 테이블은 ETF_SOR_IFO, CUS_TP_IFO
# ETF_SOR_IFO와 CUS_TP_IFO에서 제공하는 날짜가 다름
# ETF_SOR_IFO와 CUS_TP_IFO 중 겹치는 일자만 사용하기로 가정
date_list = np.array(list(set(sorted(ETF_SOR_IFO['bse_dt'].unique())) & set(sorted(CUS_TP_IFO['bse_dt'].unique()))))
date_list


# In[5]:


# ETF_SOR_IFO에서 필요한 정보만 추출
# 며칠간의 데이터를 가져올 것인지 설정
target_date = 5
ETF_SOR_IFO.sort_values(by=['etf_iem_cd', 'bse_dt'], inplace=True)
ETF_SOR_IFO_target = ETF_SOR_IFO[ETF_SOR_IFO['bse_dt'].isin(date_list[-target_date:])]

# ETF 종목 별로 지표들의 평균값을 구함
etf_score = ETF_SOR_IFO_target.groupby('etf_iem_cd').mean()
etf_score = etf_score.drop(columns=['bse_dt', 'etf_sor', 'etf_z_sor', 'z_sor_rnk'])
# yr1_tot_pft_rt가 0인 경우가 빈번히 발생 -> 상장된지 1년 안된 ETF임
etf_score


# In[6]:


# 1년 누적 수익률이 정확히 0인 값 존재 -> 상장된지 1년 안된 ETF임 -> 제외하기로 결정
etf_score[etf_score['yr1_tot_pft_rt'] == 0]


# In[7]:


# 1년 누적 수익률이 정확히 0인 값 제외
etf_infos = etf_score[etf_score['yr1_tot_pft_rt'] != 0]
etf_infos


# ## 배당금

# In[8]:


# - 값을 가지는 경우는 결측치로 판단 -> 제거
print(HISTORICAL_DIVIDEND['ddn_pym_fcy_cd'].value_counts())
HISTORICAL_DIVIDEND = HISTORICAL_DIVIDEND[HISTORICAL_DIVIDEND['ddn_pym_fcy_cd'] != '-']
HISTORICAL_DIVIDEND['ddn_pym_fcy_cd'].value_counts()


# In[9]:


HISTORICAL_DIVIDEND.sort_values(by=['etf_tck_cd', 'ediv_dt' ], inplace=True)

# 제공된 데이터는 20240528부터의 데이터 -> 이 시점의 1년 전 데이터만 반영
HISTORICAL_DIVIDEND_target = HISTORICAL_DIVIDEND[HISTORICAL_DIVIDEND['ediv_dt'] > 20230528]
HISTORICAL_DIVIDEND_target = HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'].isin(ETF_SOR_IFO['etf_iem_cd'])]
HISTORICAL_DIVIDEND_target


# In[10]:


# 배당금이랑 수정 배당금 다른 경우 존재 -> 이는 분할 or 병합이 발생한 경우로 추정
# 수정 배당금을 이용해서 배당금 관련 지표 도출
# 수정 배당금이 더 높은 경우에는 분할, 수정 배당금이 더 낮은 경우에는 병합으로 추정
(HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['ddn_amt'] != HISTORICAL_DIVIDEND_target['aed_stkp_ddn_amt']]).head()


# In[11]:


# 대상 기간 동안 배당금 주기가 변경된 ETF 확인
for i in HISTORICAL_DIVIDEND_target['etf_tck_cd'].unique():
    temp = HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i]
    temp['ddn_pym_fcy_cd'].value_counts()[0]
    if temp['ddn_pym_fcy_cd'].value_counts()[0] != len(temp):
        print(i)


# In[12]:


# 특정 주기마다 받다가 그 외의 배당금을 받는 경우가 other로 처리된다고 판단 -> other은 제외
HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == 'QQQ']


# In[13]:


for i in HISTORICAL_DIVIDEND_target['etf_tck_cd'].unique():
    HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i] = HISTORICAL_DIVIDEND_target[(HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i) & (HISTORICAL_DIVIDEND_target['ddn_pym_fcy_cd'] != 'Other')]
HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == 'QQQ']


# In[14]:


# 해다다
# HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == 'RSHO']
# for i in HISTORICAL_DIVIDEND_target['etf_tck_cd'].unique():
#     temp = HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i]
#     if (len(temp) == 1 and temp['ddn_pym_fcy_cd'].unique()[0] == 'Other'):
#         print(i, 'dfdf')
#         continue
#     else:
#         df_len = len(HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i])
#         HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i] = HISTORICAL_DIVIDEND_target[(HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i) & (HISTORICAL_DIVIDEND_target['ddn_pym_fcy_cd'] != 'Other')]
#         if len(HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i]) != df_len:
#             print(i)


# In[15]:


# groupby 후 배당금 주기 값은 최빈값, 배당금액은 평균값으로 처리
ddn_pym_fcy_cd_df = HISTORICAL_DIVIDEND_target.groupby('etf_tck_cd')['ddn_pym_fcy_cd'].agg(lambda x:x.value_counts().index[0])
aed_stkp_ddn_amt_df = HISTORICAL_DIVIDEND_target.groupby('etf_tck_cd')['aed_stkp_ddn_amt'].mean()
dividend_df = pd.merge(ddn_pym_fcy_cd_df, aed_stkp_ddn_amt_df, on='etf_tck_cd')
dividend_df


# In[16]:


def get_value(x):
    if x == 'Annual':
        return 1
    elif x == 'SemiAnnual':
        return 2
    elif x == 'Quarterly':
        return 4
    elif x == 'Monthly':
        return 12 
    # elif x == 'Other':
    #     return 12 
    else:
        return 0
dividend_df['dividend_num'] = dividend_df['ddn_pym_fcy_cd'].apply(get_value)
dividend_df['dividend_total'] = dividend_df['dividend_num'] * dividend_df['aed_stkp_ddn_amt']
dividend_df


# In[17]:


# 클러스터링에 활용될 데이터를 etf_infos로 명명
# etf score 컬럼들과 배당금 관련 컬럼들을 merge
etf_infos = pd.merge(etf_infos, dividend_df[['dividend_num','dividend_total']], left_index=True, right_index=True, how='left')
etf_infos


# In[18]:


# 배당금 컬럼의 결측치는 배당금이 없는 ETF에 대해서 발생 -> 0으로 처리
etf_infos.info()


# In[19]:


etf_infos['dividend_num'].fillna(0, inplace=True)
etf_infos['dividend_total'].fillna(0, inplace=True)


# In[20]:


etf_infos.info()


# ## 고객보유정보

# In[21]:


CUS_TP_IFO


# In[22]:


CUS_TP_IFO.sort_values(by=['tck_iem_cd', 'bse_dt' ], inplace=True)
CUS_TP_IFO_target = CUS_TP_IFO[CUS_TP_IFO['bse_dt'].isin(date_list[-target_date:])]
CUS_TP_IFO_target = CUS_TP_IFO_target[CUS_TP_IFO_target['tck_iem_cd'].isin(ETF_SOR_IFO['etf_iem_cd'])]
CUS_TP_IFO_target


# In[23]:


# 고객 정보 데이터 중 특정 범주가 100이면 다른 범주가 표시 안되는 문제 발생 -> 0으로 표시하는 처리 추가 필요
CUS_TP_IFO_target[CUS_TP_IFO_target['tck_iem_cd'] == 'XVV']


# In[24]:


cd_list = CUS_TP_IFO_target['cus_cgr_mlf_cd'].unique()
CUS_TP_IFO_target.reset_index(drop=True, inplace=True)
for i in CUS_TP_IFO_target['tck_iem_cd'].unique():
    for date in CUS_TP_IFO_target['bse_dt'].unique():
        target_cd = CUS_TP_IFO_target[(CUS_TP_IFO_target['tck_iem_cd'] == i) & (CUS_TP_IFO_target['bse_dt'] == date)]['cus_cgr_mlf_cd']
        not_cd = list(set(cd_list) - set(target_cd))
        for cd in not_cd:
            CUS_TP_IFO_target.loc[len(CUS_TP_IFO_target), :] = [date, i, int(str(cd)[0]), cd, 0, 0]
CUS_TP_IFO_target

CUS_TP_IFO_target.sort_values(by=['tck_iem_cd', 'bse_dt'], inplace=True)
CUS_TP_IFO_target['cus_cgr_llf_cd'] = CUS_TP_IFO_target['cus_cgr_llf_cd'].astype(int)
CUS_TP_IFO_target['cus_cgr_mlf_cd'] = CUS_TP_IFO_target['cus_cgr_mlf_cd'].astype(int)
CUS_TP_IFO_target['bse_dt'] = CUS_TP_IFO_target['bse_dt'].astype(int)
CUS_TP_IFO_target.reset_index(drop=True, inplace=True)
CUS_TP_IFO_target


# In[25]:


# 날짜 별로 평균값을 구함
CUS_TP_IFO_target.groupby(['tck_iem_cd', 'cus_cgr_mlf_cd'])[['cus_cgr_act_cnt_rt']].mean()


# In[26]:


temp =CUS_TP_IFO_target.groupby(['tck_iem_cd', 'cus_cgr_mlf_cd'])[['cus_cgr_act_cnt_rt']].mean().reset_index(level='cus_cgr_mlf_cd')
temp[temp['cus_cgr_mlf_cd'] == 11]


# In[27]:


# 분류가 11, 12인 경우 하나만 있어도 무방하다 판단 -> 11을 제거
temp = temp[temp['cus_cgr_mlf_cd'] != 11]
temp


# In[28]:


temp['cus_cgr_mlf_cd'].unique()


# In[30]:


for cd in temp['cus_cgr_mlf_cd'].unique():
    etf_infos = pd.merge(etf_infos, temp[temp['cus_cgr_mlf_cd'] == cd][['cus_cgr_act_cnt_rt']], left_index=True, right_index=True, how='inner')
    etf_infos.rename(columns={'cus_cgr_act_cnt_rt': 'cus_cgr_act_cnt_rt_'+str(cd)}, inplace=True)
etf_infos


# ## ETF구성종목정보

# In[31]:


temp = pd.DataFrame(columns=['etf_tck_cd', 'top5_pct', 'stock_num'])
for cd in ETF_HOLDINGS['etf_tck_cd'].unique():
    top5_sum = ETF_HOLDINGS[ETF_HOLDINGS['etf_tck_cd'] == cd].sort_values(by='wht_pct', ascending=False).head()['wht_pct'].sum()
    stock_num = len(ETF_HOLDINGS[ETF_HOLDINGS['etf_tck_cd'] == cd])
    temp.loc[len(temp), :] = [cd, top5_sum, stock_num]
temp.set_index('etf_tck_cd', inplace=True)
temp


# In[32]:


etf_infos = pd.merge(etf_infos, temp, left_index=True, right_index=True, how='inner')
etf_infos.sort_index(inplace=True)
etf_infos


# In[35]:


etf_infos.to_csv('./data/etf_infos_5일.csv', index=True)


# In[36]:


pd.read_csv('./data/etf_infos_5일.csv')


# In[ ]:





# In[ ]:




