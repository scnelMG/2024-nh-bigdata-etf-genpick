# -*- coding: utf-8 -*-

""" 0. 필요 라이브러리 import """
print('0. 필요 라이브러리 import')

import warnings
warnings.filterwarnings("ignore")

# 데이터 처리
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler

# 시각화
from sklearn.manifold import TSNE

# 클러스터링
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift
from yellowbrick.cluster import KElbowVisualizer

# 평가 지표
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 텍스트 처리
from sklearn.feature_extraction.text import TfidfVectorizer
import re
# import nltk
# 필요한 NLTK 데이터를 다운로드합니다.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag, word_tokenize

# 모델 및 해석
from xgboost import XGBRegressor, XGBClassifier
import shap
import requests


""" 1. 군집화에 사용될 데이터 병합 """
print('1. 군집화에 사용될 데이터 병합')


''' 1-1. 데이터 병합을 위한 기본 전처리 및 대상 정의 '''
print('1-1. 데이터 병합을 위한 기본 전처리 및 대상 정의')

# 데이터 불러오기
ETF_SOR_IFO = pd.read_csv('./data/본선/NH_CONTEST_ETF_SOR_IFO.csv', encoding='cp949')
HISTORICAL_DIVIDEND = pd.read_csv('./data/본선/NH_CONTEST_DATA_HISTORICAL_DIVIDEND.csv', encoding='cp949')
CUS_TP_IFO = pd.read_csv('./data/본선/NH_CONTEST_NHDATA_CUS_TP_IFO.csv', encoding='cp949')
ETF_HOLDINGS = pd.read_csv('./data/본선/NH_CONTEST_DATA_ETF_HOLDINGS.csv', encoding='cp949')

# 종목 코드(etf_iem_cd 컬럼) 뒤 공백이 있는 경우 존재 -> 제거 진행
ETF_SOR_IFO['etf_iem_cd'] = ETF_SOR_IFO['etf_iem_cd'].str.rstrip()
CUS_TP_IFO['tck_iem_cd'] = CUS_TP_IFO['tck_iem_cd'].str.rstrip()

# 사용할 날짜 정의
# 날짜 컬럼(bse_dt 컬럼)은 ETF_SOR_IFO, CUS_TP_IFO에 존재
# ETF_SOR_IFO와 CUS_TP_IFO에서 제공하는 날짜가 다름
# -> ETF_SOR_IFO와 CUS_TP_IFO 중 겹치는 일자만 사용하기로 가정
# 최종 사용할 날짜는 20240528 ~ 20240826(61일)
date_list = np.array(sorted(set(ETF_SOR_IFO['bse_dt'].unique()) & set(CUS_TP_IFO['bse_dt'].unique())))

# 며칠간의 데이터를 이용해서 군집화를 진행할 것인지 정의
# 해당 서비스의 이용자가 1일, 5일, 10일, 30일, 60일 중 선택할 수 있도록 제공할 계획
# 해당 코드는 5일을 기준으로 구현
# target_date 변수를 바꾸면 다른 날짜로도 적용 가능
target_date = 5


''' 1-2. ETF_SOR_IFO (ETF점수정보) 데이터 전처리 및 병합 '''
print('1-2. ETF_SOR_IFO (ETF점수정보) 데이터 전처리 및 병합')

# 대상 날짜에 해당하는 데이터만 추출
ETF_SOR_IFO.sort_values(by=['etf_iem_cd', 'bse_dt'], inplace=True)
ETF_SOR_IFO_target = ETF_SOR_IFO[ETF_SOR_IFO['bse_dt'].isin(date_list[-target_date:])]

# ETF 종목 별로 지표들의 평균값을 구함
etf_score = ETF_SOR_IFO_target.groupby('etf_iem_cd').mean()
etf_score = etf_score.drop(columns=['bse_dt', 'etf_sor', 'etf_z_sor', 'z_sor_rnk'])
# 1년 누적 수익률이 정확히 0인 값 존재 -> 상장된지 1년 안된 ETF임 -> 제외하기로 결정
# etf_infos : 최종적으로 군집화에 사용할 데이터프레임으로 정의
etf_infos = etf_score[etf_score['yr1_tot_pft_rt'] != 0]


''' 1-3. HISTORICAL_DIVIDEND (ETF배당내역) 데이터 전처리 및 병합 '''
print('1-3. HISTORICAL_DIVIDEND (ETF배당내역) 데이터 전처리 및 병합')

# - 값을 가지는 경우는 결측치로 판단(전체 23218개 중 19개) -> 제거
HISTORICAL_DIVIDEND = HISTORICAL_DIVIDEND[HISTORICAL_DIVIDEND['ddn_pym_fcy_cd'] != '-']
HISTORICAL_DIVIDEND.sort_values(by=['etf_tck_cd', 'ediv_dt' ], inplace=True)

# 2021년부터의 배당금 데이터가 존재 -> 모든 데이터를 사용하기 보다는 최근 1년 데이터를 사용하기로 결정
# 위에서 정의한 날짜(date_list)는 20240528부터 시작 -> 이 시점의 1년 전 데이터만 반영하기로 결정
HISTORICAL_DIVIDEND_target = HISTORICAL_DIVIDEND[HISTORICAL_DIVIDEND['ediv_dt'] > 20230528]
# 각 데이터 중 겹치는 ETF만 최종적으로 사용 -> etf_infos 데이터의 ETF 종목과 겹치는 ETF만 사용
HISTORICAL_DIVIDEND_target = HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'].isin(etf_infos.index)]

# APLY, QQQ 등 대상 기간 동안 배당금 주기가 변경된 ETF 존재
# 주기적 배당금 외의 배당금을 받는 경우가 other로 처리된다고 판단 -> other은 제외하기로 결정
for i in HISTORICAL_DIVIDEND_target['etf_tck_cd'].unique():
    HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i] = HISTORICAL_DIVIDEND_target[(HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i) & (HISTORICAL_DIVIDEND_target['ddn_pym_fcy_cd'] != 'Other')]

# 배당금(ddn_amt)이랑 수정 배당금(aed_stkp_ddn_amt) 다른 경우 존재 -> 이는 분할 or 병합이 발생한 경우로 추정
# 수정 배당금을 이용해서 배당금 관련 지표 도출
# 수정 배당금이 더 높은 경우에는 분할, 수정 배당금이 더 낮은 경우에는 병합으로 추정
# 종목 코드(etf_tck_cd) 기준으로 groupby 후 배당금 주기 값은 최빈값, 배당금액은 평균값으로 처리
ddn_pym_fcy_cd_df = HISTORICAL_DIVIDEND_target.groupby('etf_tck_cd')['ddn_pym_fcy_cd'].agg(lambda x:x.value_counts().index[0])
aed_stkp_ddn_amt_df = HISTORICAL_DIVIDEND_target.groupby('etf_tck_cd')['aed_stkp_ddn_amt'].mean()
dividend_df = pd.merge(ddn_pym_fcy_cd_df, aed_stkp_ddn_amt_df, on='etf_tck_cd')

# 주기 컬럼은 1년을 기준으로 몇 번 배당금을 받을 수 있는지로 변경
## ex) 분기(Quarterly) -> 1년에 4번
def get_value(x):
    if x == 'Annual':
        return 1
    elif x == 'SemiAnnual':
        return 2
    elif x == 'Quarterly':
        return 4
    elif x == 'Monthly':
        return 12   
    else:
        return 0
dividend_df['dividend_num'] = dividend_df['ddn_pym_fcy_cd'].apply(get_value)
dividend_df['dividend_total'] = dividend_df['dividend_num'] * dividend_df['aed_stkp_ddn_amt']

# etf_infos에 배당금 관련 컬럼들을 병합
# etf_infos에는 있는데 배당금 데이터에는 없는 ETF의 경우는 배당이 안나오는 ETF로 확인
# -> 배당금이 안나오는 ETF는 0으로 처리
etf_infos = pd.merge(etf_infos, dividend_df[['dividend_num','dividend_total']], left_index=True, right_index=True, how='left')
etf_infos['dividend_num'].fillna(0, inplace=True)
etf_infos['dividend_total'].fillna(0, inplace=True)


''' 1-4. CUS_TP_IFO (고객보유정보) 데이터 전처리 및 병합 '''
print('1-4. CUS_TP_IFO (고객보유정보) 데이터 전처리 및 병합')

# 대상 날짜에 해당하는 데이터만 추출
CUS_TP_IFO.sort_values(by=['tck_iem_cd', 'bse_dt'], inplace=True)
CUS_TP_IFO_target = CUS_TP_IFO[CUS_TP_IFO['bse_dt'].isin(date_list[-target_date:])]

# 고객보유정보 중 최신 날짜의 데이터가 존재하지 않은 경우 존재
# 해당 경우에 속하는 ETF(IYG, DON 등)의 정보를 나무 증권의 'NH데이터'정보에서 투자자수 부족의 이유로 제공 안함
# -> 해당 경우의 ETF는 제외하기로 결정
CUS_TP_IFO_target = CUS_TP_IFO_target[CUS_TP_IFO_target['tck_iem_cd'].isin(CUS_TP_IFO[CUS_TP_IFO['bse_dt'].isin(date_list[-1:])]['tck_iem_cd'].unique())]

# 각 데이터 중 겹치는 ETF만 최종적으로 사용 -> etf_infos 데이터의 ETF 종목과 겹치는 ETF만 사용
CUS_TP_IFO_target = CUS_TP_IFO_target[CUS_TP_IFO_target['tck_iem_cd'].isin(etf_infos.index)]

# 고객 정보 데이터 중 특정 중분류코드에 대한 데이터가 없는 경우 존재
## ex1) 11 범주가 100이면 12 범주는 안나타나있음
## ex2) 21 범주가 70이고 22 범주가 30이면 23, 24, 25 범주는 안나타나있음
# -> 해당 데이터는 다른 중분류코드들의 합이 100이 되어버려 표시가 안되고 있음을 확인
# -> 제외된 중분류코드는 0으로 설정하고 추가하기로 결정

cd_list = CUS_TP_IFO_target['cus_cgr_mlf_cd'].unique()
CUS_TP_IFO_target.reset_index(drop=True, inplace=True)
for i in CUS_TP_IFO_target['tck_iem_cd'].unique():
    for date in CUS_TP_IFO_target['bse_dt'].unique():
        target_cd = CUS_TP_IFO_target[(CUS_TP_IFO_target['tck_iem_cd'] == i) & (CUS_TP_IFO_target['bse_dt'] == date)]['cus_cgr_mlf_cd']
        # 특정 날짜에 존재하지 않는 고객분류코드 파악
        not_cd = list(set(cd_list) - set(target_cd))
        # 존재하지 않는 코드에 대해서 새로운 열 추가
        for cd in not_cd:
            CUS_TP_IFO_target.loc[len(CUS_TP_IFO_target), :] = [date, i, int(str(cd)[0]), cd, 0, 0]

# 위 처리로 열의 type이 변경되어 원래대로 변경
CUS_TP_IFO_target.sort_values(by=['tck_iem_cd', 'bse_dt'], inplace=True)
CUS_TP_IFO_target['cus_cgr_llf_cd'] = CUS_TP_IFO_target['cus_cgr_llf_cd'].astype(int)
CUS_TP_IFO_target['cus_cgr_mlf_cd'] = CUS_TP_IFO_target['cus_cgr_mlf_cd'].astype(int)
CUS_TP_IFO_target['bse_dt'] = CUS_TP_IFO_target['bse_dt'].astype(int)
CUS_TP_IFO_target.reset_index(drop=True, inplace=True)

# 고객정보 데이터에 비율이 두 가지 제공 됨 : 고객구성계좌수비율, 고객구성투자비율
# 고객구성투자비율은 금액의 비율이기 때문에, 자산이 많은 경우 왜곡될 가능성이 있다고 판단
# 고객구성계좌수비율은 그러한 왜곡이 일어날 가능성이 낮다고 판단해 고객구성계좌수비율만 사용하기로 결정
CUS_TP_IFO_target_means = CUS_TP_IFO_target.groupby(['tck_iem_cd', 'cus_cgr_mlf_cd'])[['cus_cgr_act_cnt_rt']].mean().reset_index(level='cus_cgr_mlf_cd')

# 분류가 11, 12인 경우 하나만 있어도 무방 -> 11을 제거
## ex) 11 범주가 70이면 자동으로 12는 30로 결정되기 때문
CUS_TP_IFO_target_means = CUS_TP_IFO_target_means[CUS_TP_IFO_target_means['cus_cgr_mlf_cd'] != 11]

# 각 중분류코드 별로 etf_infos에 병합
for cd in CUS_TP_IFO_target_means['cus_cgr_mlf_cd'].unique():
    etf_infos = pd.merge(etf_infos, CUS_TP_IFO_target_means[CUS_TP_IFO_target_means['cus_cgr_mlf_cd'] == cd][['cus_cgr_act_cnt_rt']], left_index=True, right_index=True, how='inner')
    etf_infos.rename(columns={'cus_cgr_act_cnt_rt': 'cus_cgr_act_cnt_rt_'+str(cd)}, inplace=True)


''' 1-5. ETF_HOLDINGS (ETF구성종목정보) 데이터 전처리 및 병합 '''
print('1-5. ETF_HOLDINGS (ETF구성종목정보) 데이터 전처리 및 병합')

# ETF 구성종목 중 가장 비중이 높은 5개 종목의 비중 합과 전체 구성 종목 수 도출
ETF_top_ratio = pd.DataFrame(columns=['etf_tck_cd', 'top5_pct'])
for cd in ETF_HOLDINGS['etf_tck_cd'].unique():
    top5_sum = ETF_HOLDINGS[ETF_HOLDINGS['etf_tck_cd'] == cd].sort_values(by='wht_pct', ascending=False).head(5)['wht_pct'].sum()
    ETF_top_ratio.loc[len(ETF_top_ratio), :] = [cd, top5_sum]
ETF_top_ratio.set_index('etf_tck_cd', inplace=True)

etf_infos = pd.merge(etf_infos, ETF_top_ratio, left_index=True, right_index=True, how='inner')
etf_infos.sort_index(inplace=True)


''' 1-6. 반영 날짜 수에 따른 군집화 대상 데이터 도출 함수 정의 '''
print('1-6. 반영 날짜 수에 따른 군집화 대상 데이터 도출 함수 정의')

def make_data_for_clustering(target_date, date_list):
    print('데이터 불러오기')
    ETF_SOR_IFO = pd.read_csv('./data/본선/NH_CONTEST_ETF_SOR_IFO.csv', encoding='cp949')
    HISTORICAL_DIVIDEND = pd.read_csv('./data/본선/NH_CONTEST_DATA_HISTORICAL_DIVIDEND.csv', encoding='cp949')
    CUS_TP_IFO = pd.read_csv('./data/본선/NH_CONTEST_NHDATA_CUS_TP_IFO.csv', encoding='cp949')
    ETF_HOLDINGS = pd.read_csv('./data/본선/NH_CONTEST_DATA_ETF_HOLDINGS.csv', encoding='cp949')
    ETF_SOR_IFO['etf_iem_cd'] = ETF_SOR_IFO['etf_iem_cd'].str.rstrip()
    CUS_TP_IFO['tck_iem_cd'] = CUS_TP_IFO['tck_iem_cd'].str.rstrip()

    print('ETF score 데이터 추출')
    ETF_SOR_IFO.sort_values(by=['etf_iem_cd', 'bse_dt'], inplace=True)
    ETF_SOR_IFO_target = ETF_SOR_IFO[ETF_SOR_IFO['bse_dt'].isin(date_list[-target_date:])]
    # ETF 종목 별로 지표들의 평균값을 구함
    etf_score = ETF_SOR_IFO_target.groupby('etf_iem_cd').mean()
    etf_score = etf_score.drop(columns=['bse_dt', 'etf_sor', 'etf_z_sor', 'z_sor_rnk'])
    etf_infos = etf_score[etf_score['yr1_tot_pft_rt'] != 0]

    print('배당금 데이터 추출')
    HISTORICAL_DIVIDEND = HISTORICAL_DIVIDEND[HISTORICAL_DIVIDEND['ddn_pym_fcy_cd'] != '-']
    HISTORICAL_DIVIDEND.sort_values(by=['etf_tck_cd', 'ediv_dt' ], inplace=True)
    # 제공된 데이터는 20240528부터의 데이터 -> 이 시점의 1년 전 데이터만 반영
    HISTORICAL_DIVIDEND_target = HISTORICAL_DIVIDEND[HISTORICAL_DIVIDEND['ediv_dt'] > 20230528]
    HISTORICAL_DIVIDEND_target = HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'].isin(ETF_SOR_IFO['etf_iem_cd'])]
    for i in HISTORICAL_DIVIDEND_target['etf_tck_cd'].unique():
        HISTORICAL_DIVIDEND_target[HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i] = HISTORICAL_DIVIDEND_target[(HISTORICAL_DIVIDEND_target['etf_tck_cd'] == i) & (HISTORICAL_DIVIDEND_target['ddn_pym_fcy_cd'] != 'Other')]
    # groupby 후 배당금 주기 값은 최빈값, 배당금액은 평균값으로 처리
    ddn_pym_fcy_cd_df = HISTORICAL_DIVIDEND_target.groupby('etf_tck_cd')['ddn_pym_fcy_cd'].agg(lambda x:x.value_counts().index[0])
    aed_stkp_ddn_amt_df = HISTORICAL_DIVIDEND_target.groupby('etf_tck_cd')['aed_stkp_ddn_amt'].mean()
    dividend_df = pd.merge(ddn_pym_fcy_cd_df, aed_stkp_ddn_amt_df, on='etf_tck_cd')
    dividend_df['dividend_num'] = dividend_df['ddn_pym_fcy_cd'].apply(get_value)
    dividend_df['dividend_total'] = dividend_df['dividend_num'] * dividend_df['aed_stkp_ddn_amt']
    etf_infos = pd.merge(etf_infos, dividend_df[['dividend_num','dividend_total']], left_index=True, right_index=True, how='left')
    etf_infos['dividend_num'].fillna(0, inplace=True)
    etf_infos['dividend_total'].fillna(0, inplace=True)

    print('고객 정보 데이터 추출')
    CUS_TP_IFO.sort_values(by=['tck_iem_cd', 'bse_dt' ], inplace=True)
    CUS_TP_IFO_target = CUS_TP_IFO[CUS_TP_IFO['bse_dt'].isin(date_list[-target_date:])]
    CUS_TP_IFO_target = CUS_TP_IFO_target[CUS_TP_IFO_target['tck_iem_cd'].isin(CUS_TP_IFO[CUS_TP_IFO['bse_dt'].isin(date_list[-1:])]['tck_iem_cd'].unique())]
    CUS_TP_IFO_target = CUS_TP_IFO_target[CUS_TP_IFO_target['tck_iem_cd'].isin(ETF_SOR_IFO['etf_iem_cd'])]
    cd_list = CUS_TP_IFO_target['cus_cgr_mlf_cd'].unique()
    CUS_TP_IFO_target.reset_index(drop=True, inplace=True)
    for i in CUS_TP_IFO_target['tck_iem_cd'].unique():
        for date in CUS_TP_IFO_target['bse_dt'].unique():
            target_cd = CUS_TP_IFO_target[(CUS_TP_IFO_target['tck_iem_cd'] == i) & (CUS_TP_IFO_target['bse_dt'] == date)]['cus_cgr_mlf_cd']
            not_cd = list(set(cd_list) - set(target_cd))
            for cd in not_cd:
                CUS_TP_IFO_target.loc[len(CUS_TP_IFO_target), :] = [date, i, int(str(cd)[0]), cd, 0, 0]
    CUS_TP_IFO_target.sort_values(by=['tck_iem_cd', 'bse_dt'], inplace=True)
    CUS_TP_IFO_target['cus_cgr_llf_cd'] = CUS_TP_IFO_target['cus_cgr_llf_cd'].astype(int)
    CUS_TP_IFO_target['cus_cgr_mlf_cd'] = CUS_TP_IFO_target['cus_cgr_mlf_cd'].astype(int)
    CUS_TP_IFO_target['bse_dt'] = CUS_TP_IFO_target['bse_dt'].astype(int)
    CUS_TP_IFO_target.reset_index(drop=True, inplace=True)
    CUS_TP_IFO_target_means =CUS_TP_IFO_target.groupby(['tck_iem_cd', 'cus_cgr_mlf_cd'])[['cus_cgr_act_cnt_rt']].mean().reset_index(level='cus_cgr_mlf_cd')
    CUS_TP_IFO_target_means = CUS_TP_IFO_target_means[CUS_TP_IFO_target_means['cus_cgr_mlf_cd'] != 11]
    for cd in CUS_TP_IFO_target_means['cus_cgr_mlf_cd'].unique():
        etf_infos = pd.merge(etf_infos, CUS_TP_IFO_target_means[CUS_TP_IFO_target_means['cus_cgr_mlf_cd'] == cd][['cus_cgr_act_cnt_rt']], left_index=True, right_index=True, how='inner')
        etf_infos.rename(columns={'cus_cgr_act_cnt_rt': 'cus_cgr_act_cnt_rt_'+str(cd)}, inplace=True)

    print('ETF 보유 종목 데이터 추출')
    ETF_top_ratio = pd.DataFrame(columns=['etf_tck_cd', 'top5_pct'])
    for cd in ETF_HOLDINGS['etf_tck_cd'].unique():
        top5_sum = ETF_HOLDINGS[ETF_HOLDINGS['etf_tck_cd'] == cd].sort_values(by='wht_pct', ascending=False).head()['wht_pct'].sum()
        ETF_top_ratio.loc[len(ETF_top_ratio), :] = [cd, top5_sum]
    ETF_top_ratio.set_index('etf_tck_cd', inplace=True)
    etf_infos = pd.merge(etf_infos, ETF_top_ratio, left_index=True, right_index=True, how='inner')
    etf_infos.sort_index(inplace=True)

    return etf_infos

# 1, 5, 10, 30, 60일 데이터 별로 결과를 구현하기로 함
# 1, 5, 10, 30, 60일 데이터 셋 생성하는 반복문 -> 해당 코드에서는 실행 안함 -> 주석처리

# ETF_SOR_IFO = pd.read_csv('./data/본선/NH_CONTEST_ETF_SOR_IFO.csv', encoding='cp949')
# CUS_TP_IFO = pd.read_csv('./data/본선/NH_CONTEST_NHDATA_CUS_TP_IFO.csv', encoding='cp949')
# date_list = np.array(sorted(set(ETF_SOR_IFO['bse_dt'].unique()) and set(CUS_TP_IFO['bse_dt'].unique())))
# for days in [1, 5, 10, 30, 60]:
#     print("대상 일자 : ", days)
#     etf_infos = make_data_for_clustering(days, date_list)
#     etf_infos.to_csv(f'./data/etf_infos_{days}일.csv', index=True)


""" 2. ETF 관련 데이터들을 이용한 군집화 진행 """
print('2. ETF 관련 데이터들을 이용한 군집화 진행')


''' 2-1. 군집화를 위한 데이터 전처리 '''
print('2-1. 군집화를 위한 데이터 전처리')

# 군집화를 위한 스케일링 진행 -> StandardScaler 사용
scaler = StandardScaler()
scaler.fit(etf_infos)
etf_infos_std = scaler.transform(etf_infos)
etf_infos_std = pd.DataFrame(etf_infos_std, columns=etf_infos.columns, index=etf_infos.index)

# 군집화 대상 컬럼이 많다고 판단(군집화 성능 지표도 안좋았음) -> 차원 축소 진행
# 비선형 차원 축소 기법인 t-SNE 사용
tsne = TSNE(n_components=2, random_state=42)  
etf_infos_tsne = tsne.fit_transform(etf_infos_std)


''' 2-2. 군집 방법 결정을 위한 실험 '''
print('2-2. 군집 방법 결정을 위한 실험')


''' 2-2-a. 군집 방법 별 최적의 군집 수 찾기'''
print('2-2-a. 군집 방법 별 최적의 군집 수 찾기')

# elbow method를 이용한 최적의 군집 수 찾기 -> KElbowVisualizer 함수를 사용해서 최적 군집 수 도출
# 실험 대상 군집화 알고리즘 : [K-means, 병합 군집화(agglomerative clustering), 스펙트럼 군집화(SpectralClustering), MeanShift]

# K-means 군집화 알고리즘에 대해서 최적의 군집 수 찾기
model = KMeans(random_state=42)
elbow_visualizer = KElbowVisualizer(model, k=(2, 15), metric='distortion',  timings=False)
elbow_visualizer.fit(etf_infos_tsne)
kmeans_optimal = elbow_visualizer.elbow_value_

# 병합 군집화(agglomerative clustering) 알고리즘에 대해서 최적의 군집 수 찾기
model = AgglomerativeClustering()
elbow_visualizer = KElbowVisualizer(model, k=(2, 15), metric='distortion',  timings=False)
elbow_visualizer.fit(etf_infos_tsne)
Agglomerative_optimal = elbow_visualizer.elbow_value_

# 스펙트럼 군집화(SpectralClustering) 알고리즘에 대해서 최적의 군집 수 찾기
model = SpectralClustering(affinity='nearest_neighbors', random_state=42)
elbow_visualizer = KElbowVisualizer(model, k=(2, 15), metric='distortion',  timings=False)
elbow_visualizer.fit(etf_infos_tsne)
Spectral_optimal = elbow_visualizer.elbow_value_

# MeanShift 알고리즘은 자동으로 군집수가 결정되기 때문에 따로 최적의 군집 수를 찾을 필요가 없음


''' 2-2-b. 군집 방법들 간의 성능 지표 비교'''
print('2-2-b. 군집 방법들 간의 성능 지표 비교')

# 앞서 언급한 총 4개의 군집화 방법에 대해서 성능 지표 비교 진행
# Silhouette, Calinski-Harabasz, Davies-Bouldin 세 가지 지표 사용
# Silhouette : 군집 내에서 얼마나 밀집되어 있는지와 다른 군집과 얼마나 구별되는지를 측정(0 ~ 1 사이의 값을 가지며 1에 가까울수록 군집화가 잘 되었다고 판단)
# Calinski-Harabasz : 군집 내 분산과 군집 간 분산의 비율을 측정(값이 클수록 군집화가 잘 되었다고 판단)
# Davies-Bouldin : 군집 간 분리도과 군집 내 응집도을 측정(값이 작을수록 군집화가 잘 되었다고 판단)

# 실험 진행 및 결과 저장
cluster_result = pd.DataFrame(columns=["Method", 'cluster_n', "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"])
for method in ["KMeans", "Agglomerative", "Spectral", "MeanShift"]:
    if method == "KMeans":
        model = KMeans(n_clusters=kmeans_optimal, random_state=42).fit(etf_infos_tsne)
        cluster_n = kmeans_optimal
        labels = model.labels_

    elif method == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=Agglomerative_optimal).fit(etf_infos_tsne)
        cluster_n = Agglomerative_optimal
        labels = model.labels_

    elif method == "Spectral":
        model = SpectralClustering(n_clusters=Spectral_optimal, affinity='nearest_neighbors',random_state=42).fit(etf_infos_tsne)
        cluster_n = Spectral_optimal
        labels = model.labels_

    elif method == "MeanShift":
        model = MeanShift().fit(etf_infos_tsne)
        cluster_n = len(np.unique(model.labels_))
        labels = model.labels_

    silhouette = silhouette_score(etf_infos_tsne, labels)
    calinski_harabasz = calinski_harabasz_score(etf_infos_tsne, labels) 
    davies_bouldin = davies_bouldin_score(etf_infos_tsne, labels)

    cluster_result.loc[len(cluster_result)] = [method, cluster_n, silhouette, calinski_harabasz, davies_bouldin]

# 해당 결과 K-means 알고리즘이 제일 성능이 좋았음 
# [성능 출력 결과]
## Silhouette : 0.434266, Calinski-Harabasz : 266.694501, Davies-Bouldin : 0.737051
# 군집 4개, K-means 알고리즘을 이용하여 군집화 및 군집 해석 진행
final_model = KMeans(n_clusters=kmeans_optimal, random_state=42).fit(etf_infos_tsne)

# 차원 축소하기 전 데이터에 군집 결과 추가
etf_infos_std['cluster'] = final_model.labels_


''' 2-3. 군집화 결과에 대한 해석 '''
print('2-3. 군집화 결과에 대한 해석')


''' 2-3-a. 군집화에 사용된 컬럼들의 중요도 확인 '''
print('2-3-a. 군집화에 사용된 컬럼들의 중요도 확인')

# 각 데이터가 어떤 군집에 속하는지를 종속변수 y로 설정하여 분류모델 학습
# 그 후 Feature Importance를 통해 각 컬럼의 중요도 확인

# XGBoost 모델을 이용하여 분류 모델 학습
xgb = XGBClassifier(random_state=42)
X = etf_infos_std.drop('cluster', axis=1)
y = etf_infos_std['cluster']
xgb.fit(X, y)

# Feature Importance 도출
feature_importances = xgb.feature_importances_
features = etf_infos_std.columns

# 중요도 높은 순으로 정렬
indices = np.argsort(feature_importances)[::-1]
feature_importances = feature_importances[indices]
features = [features[i] for i in indices]

# [중요도 상위 5개 컬럼 출력 결과 예시]
## acl_pft_rt_z_sor : 0.224
## trk_err_z_sor : 0.12
## shpr_z_sor : 0.119
## dividend_num : 0.095
## yr1_tot_pft_rt : 0.071


''' 2-3-b. 군집화 결과 해석 및 시각화 '''
print('2-3-b. 군집화 결과 해석 및 시각화')

# 각 군집별로 컬럼들에 대해서 boxplot을 그려서 군집 특성 파악
# 이 때 컬럼들의 중요도도 참고
# Tableau, ppt로 그래프 생성할 계획
# 해당 제출 코드에서는 그래프 생성 코드는 제외함


''' 2-4. 각 군집의 대표 ETF 도출 '''
print('2-4. 각 군집의 대표 ETF 도출')

# 군집의 대표 ETF는 각 군집의 중심에서 가까운 ETF라고 정의
# 클러스터 중심 좌표 가져오기
centroids = final_model.cluster_centers_

# 각 군집별로 중심에서 가까운 데이터 3개씩 구하기
closest_data_per_cluster = {}

output_etf_num = 5
for i in range(kmeans_optimal):
    # 군집 i에 속한 데이터의 인덱스를 찾기
    cluster_indices = np.where(final_model.labels_ == i)[0]
    cluster_data = pd.DataFrame(etf_infos_tsne).iloc[cluster_indices, :]
    
    # 클러스터 중심과의 거리 계산
    distances = np.linalg.norm(cluster_data - centroids[i], axis=1)
    
    # 거리 기준으로 가까운 3개의 데이터 선택
    closest_indices = cluster_indices[np.argsort(distances)[:output_etf_num]]
    
    # 결과 저장
    closest_data_per_cluster[i] = closest_indices

# [출력 결과 예시]
# Cluster 0의 대표 ETF 5개 : ['SCHB', 'SPYX', 'SPHQ', 'PBUS', 'SCHX']
# Cluster 1의 대표 ETF 5개 : ['TECB', 'ONEQ', 'FTEC', 'XLK', 'XHB']
# Cluster 2의 대표 ETF 5개 : ['VIOV', 'FDIS', 'VTWO', 'SPSM', 'PEJ']
# Cluster 3의 대표 ETF 5개 : ['NOBL', 'IHE', 'SCHD', 'SDY', 'OUSA']


""" 3. ETF 구성 종목들의 요약 """
print('3. ETF 구성 종목들의 요약')


''' 3-1. 요약에 필요한 데이터 불러오기 및 전처리 '''
print('3-1. 요약에 필요한 데이터 불러오기 및 전처리')


# ETF구성종목정보, 해외종목정보 테이블 데이터 불러오기
ETF_HOLDINGS = pd.read_csv('./data/NH_CONTEST_DATA_ETF_HOLDINGS.csv', encoding='cp949')
NW_FC_STK_IEM_IFO = pd.read_csv('./data/NH_CONTEST_NW_FC_STK_IEM_IFO.csv', encoding='cp949')

# 요약을 진행할 ETF 선택
target_etf = 'NOBL'

# ETF구성종목정보 테이블에서 대상 ETF(QQQ)의 구성종목, 구성 비율을 추출
ETF_HOLDINGS_target = ETF_HOLDINGS[ETF_HOLDINGS['etf_tck_cd'] == target_etf][['etf_tck_cd', 'tck_iem_cd', 'wht_pct']]

# 해외종목정보 테이블에서 종목코드, 영문사업개요내용을 추출
NW_FC_STK_IEM_IFO = NW_FC_STK_IEM_IFO[['tck_iem_cd', 'eng_utk_otl_cts']]

# 대상 ETF(QQQ)의 구성종목의 영문사업개요내용을 확인
etf_describe = pd.merge(ETF_HOLDINGS_target, NW_FC_STK_IEM_IFO, on='tck_iem_cd')
etf_describe.sort_values(by='wht_pct', ascending=False, inplace=True)

# 구성하고 있는 모든 종목들의 설명을 생성형 AI에 인풋하기에는 데이터가 너무 많음
# 생성형 AI의 인풋 토큰 제한 때문
# -> 상위 30개 종목만 선택
## 만약 30개보다 적은 경우는 모든 종목 선택
max_etf_num = 30
if etf_describe.shape[0] > max_etf_num:
    etf_describe_target = etf_describe.head(max_etf_num)
else:
    etf_describe_target = etf_describe
etf_describe_target.drop('etf_tck_cd', axis=1, inplace=True)

# 데이터프레임을 microsoft azure openai에 입력할 수 있는 형태로 변환 필요
# 데이터프레임을 딕셔너리로 변환
df_dict = etf_describe_target.to_dict(orient='records')

# 닉셔너리를 문자열로 변환
df_dict_string = '\n'.join([str(record) for record in df_dict])


''' 3-2. 생성형 AI를 이용한 요약 진행 '''
print('3-2. 생성형 AI를 이용한 요약 진행')

# gpt-4o-mini 모델 사용
# Azure openai에서 제공하는 예시 코드 활용

# 발급 받은 API KEY 입력 필요
API_KEY = "YOUR_API_KEY"
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# prompt 작성 및 ETF 구성 종목들의 설명 데이터 입력
payload = {
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "나는 너한테 dictionary 형태로 ETF에 대해 어떤 주식 종목들이 구성하고 있는지에 대한 정보를 줄거야. \n 종목들의 설명들을 종합해서 ETF에 대한 구체적인 주요 정보들을 영어로 요약해주고 그 내용 그대로 한글도 요약해줘(단,  영어 문장과 한글 문장은 줄 바꿈 해줘).\n 그리고 요약본 길이는 3 문장 이내로 하고, 내용에 종목명을 언급하지 않지만 인풋된 주식 설명(eng_utk_otl_cts)에 사용된 단어들 위주로 사용해. \n  제공하는 정보는 ETF를 구성하는 주식 종목 코드(tck_iem_cd), ETF에서의 주식 구성 비율(wht_pct), 주식 설명(eng_utk_otl_cts)을 줄거야.\n 주식 구성 비율(wht_pct)에 따라 요약본에 반영되는 정보를 달리 해줘 \n그 요약본은 ETF 큐레이션에 사용될거야 큐레이션에 필요한 정보들을 풍부하지만 간결하게 담아줘."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": df_dict_string
        }
      ]
    }
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 800
}

ENDPOINT = "https://nhpass2024.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-02-15-preview"

# 요청 보내기
try:
    response = requests.post(ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
except requests.RequestException as e:
    raise SystemExit(f"Failed to make the request. Error: {e}")

# 요약 결과 저장
# 영어 요약본 -> 추후 키워드 중요도 확인에 사용할 계획 -> 중요도가 산출되는 키워드가 영어이기 때문
en_summation = response.json()['choices'][0]['message']['content'].split('\n')[0]
# 한글 용약본 -> 사용자에게 직접 제공되는 요약본
ko_summation = response.json()['choices'][0]['message']['content'].split('\n')[-1]

# [요약 결과 출력 예시]
# print('영어 요약본 :', en_summation)
# 영어 요약본 : This ETF comprises a diverse range of companies focused on consumer health, industrial tools, logistics, biopharmaceuticals, and food distribution. With a strong emphasis on global reach and a wide array of product offerings, these firms operate in various segments such as healthcare, food and beverage, and industrial services. The portfolio reflects significantion from essential health solutions, consumer goods, and innovative technology in their respective industries.
# print('한글 요약본 :', ko_summation)
# 한글 요약본 : 이 ETF는 소비자 건강, 산업 도구, 물류, 생명공학 및 식품 유통에 중점을 둔 다양한 기업들로 구성되어 있습니다. 글로벌 시장과 폭넓은 제품군에 중점을 두고 있으며, 헬스케어, 식음료, 산업 서비스 등 다양한 분야에서 활동하는 기업들이 포함되어 있습니다. 포트폴리오는 필수 건강 솔루션, 소비재 및 각 산업에서의 혁신 기술이 두드러진 비중을 반영합니다.


""" 4. 구성 종목들의 설명과 수익률 간의 키워드 중요도 추출 """
print('4. 구성 종목들의 설명과 수익률 간의 키워드 중요도 추출')


''' 4-1. 구성 종목 설명 데이터 불러오기 및 전처리 '''
print('4-1. 구성 종목 설명 데이터 불러오기 및 전처리')

# 데이터 불러오기
NW_FC_STK_IEM_IFO = pd.read_csv('./data/NH_CONTEST_NW_FC_STK_IEM_IFO.csv', encoding='cp949')

# ETF인 경우 제외 추출
NW_FC_STK_IEM_IFO = NW_FC_STK_IEM_IFO[NW_FC_STK_IEM_IFO['stk_etf_dit_cd'] != 'ETF']

# 영문사업개요내용(ENG_UTK_OTL_CTS)의 길이 열 추가
NW_FC_STK_IEM_IFO.loc[:, 'eng_utk_otl_cts_len'] = NW_FC_STK_IEM_IFO['eng_utk_otl_cts'].apply(lambda x: len(x))
# 영문사업개요내용이 없는 행 제거 
NW_FC_STK_IEM_IFO = NW_FC_STK_IEM_IFO[NW_FC_STK_IEM_IFO['eng_utk_otl_cts_len'] > 1]

# 전처리 함수 정의
def preprocess_text(text):
    # 소문자 변환
    text = text.lower()
    # 영어 외 다른 문자 제거
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 구두점 제거
    text = re.sub(r'[^\w\s]', '', text)
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    # 종목 코드와 종목명 불용어에 추가
    stop_words = stop_words.union(set(NW_FC_STK_IEM_IFO['tck_iem_cd'].apply(lambda x: x.lower()).values))
    stop_words = stop_words.union(set(NW_FC_STK_IEM_IFO['fc_sec_eng_nm'].apply(lambda x: x.lower()).str.split().sum()))
    
    # 품사 태깅 후 명사와 동사만 선택
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    filtered_words = [word for word, pos in tagged_words if pos.startswith('N') or pos.startswith('V')]

    # 불용어 제거 및 어간 추출
    filtered_words = [word for word in filtered_words if word not in stop_words]
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in filtered_words])
    
    return text


# 정의된 함수를 이용해 영문사업개요내용 열 전처리
NW_FC_STK_IEM_IFO['eng_utk_otl_cts_pro'] = NW_FC_STK_IEM_IFO['eng_utk_otl_cts'].apply(preprocess_text)


''' 4-2. 수익률 데이터 불러오기 및 전처리 '''
print('4-2. 수익률 데이터 불러오기 및 전처리')

# 주식일별정보 테이블 불러오기
STK_DD_IFO = pd.read_csv('./data/NH_CONTEST_NHDATA_STK_DD_IFO.csv', encoding='cp949')

# 종목 코드의 오른쪽 공백 제거
STK_DD_IFO['tck_iem_cd'] = STK_DD_IFO['tck_iem_cd'].str.rstrip()

# 날짜 기준으로 정렬
STK_DD_IFO.sort_values(by=['tck_iem_cd', 'bse_dt'], inplace=True)
STK_DD_IFO.reset_index(drop=True, inplace=True)

# 주식일별정보 테이블에서 최근 특정 기간 간의 평균 수익률 계산
# 며칠간의 데이터를 이용해서 수익률을 계산할 것인지 정의
# 해당 서비스의 이용자가 1일, 5일, 10일, 30일, 60일 중 선택할 수 있도록 제공할 계획
# 해당 코드는 5일을 기준으로 구현
# target_date 변수를 바꾸면 다른 날짜로도 적용 가능
target_date_earn = 5
date_list_earn = np.array(sorted(STK_DD_IFO['bse_dt'].unique()))
earn_df = STK_DD_IFO[STK_DD_IFO['bse_dt'].isin(date_list_earn[-target_date_earn:])].groupby('tck_iem_cd').mean()[['tco_avg_pft_rt']]

# 영문사업개요내용과 5영업일 간의 평규 수익률을 조인
earn_description_df = pd.merge(NW_FC_STK_IEM_IFO, earn_df, left_on='tck_iem_cd', right_on= earn_df.index)


''' 4-3 영문사업개요내용과 수익률 간의 관계를 파악하기 위한 모델 생성 '''
print('4-3 영문사업개요내용과 수익률 간의 관계를 파악하기 위한 모델 생성')

# TF-IDF 벡터라이저 생성
tfidf_vectorizer = TfidfVectorizer(max_features=100)

# 전처리된 영문사업개요내용 열에 TF-IDF 적용
tfidf_matrix = tfidf_vectorizer.fit_transform(earn_description_df['eng_utk_otl_cts_pro'])

# 결과를 데이터프레임으로 변환
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# tfidf_df에 종목코드, 평균 수익률 추가
tfidf_df['tck_iem_cd'] = earn_description_df['tck_iem_cd']
tfidf_df['tco_avg_pft_rt'] = earn_description_df['tco_avg_pft_rt']

# tfidf 결과로 대상 간의 평균 수익률을 예측하는 회귀 모델 생성
X = tfidf_df.drop(['tck_iem_cd', 'tco_avg_pft_rt'], axis=1)
y = tfidf_df['tco_avg_pft_rt']

# XGBoost 회귀 모델 생성 및 훈련
xgb_reg = XGBRegressor(random_state=42)
xgb_reg.fit(X, y)


''' 4-4 SHAP를 이용한 중요도 도출 '''
print('4-4 SHAP를 이용한 중요도 도출')

# SHAP 모델 생성 후 shap value 계산
explainer = shap.Explainer(xgb_reg)
shap_values = explainer(X)

# 단어별 중요도 데이터프레임 생성
shap_values = explainer.shap_values(X)
shap_abs_mean = np.mean(np.abs(shap_values), axis=0)

importance_df = pd.DataFrame({
    'feature': X.columns, 
    'importance': shap_abs_mean  
})

# 요약본 속 들어있는 키워드 중요도 출력
# for i in importance_df[importance_df['feature'].isin(en_summation.split())].values:
#     print(i[0], round(i[1], 3))
#
# 해당 부분을 한글로 변환하는 작업 필요
# 영어 요약본 : This ETF comprises a diverse range of companies engaged in consumer health, industrial tools, logistics, biopharmaceuticals, cleaning products, and food distribution. With a focus on essential health solutions, innovative tools, and logistics services, the portfolio showcases strong global presence and market adaptability. The companies are well-positioned within their respective industries, contributing to a robust and balanced investment strategy.
# [요약본 속 들어있는 키워드 중요도 출력 예시]
# portfolio 0.287
# product 0.941