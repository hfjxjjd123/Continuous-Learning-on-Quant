# Trading with the Momentum Transformer
## About
This code accompanies the paper [Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture](https://arxiv.org/pdf/2112.08534.pdf) and additionally provides an implementation for the paper [Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection](https://arxiv.org/pdf/2105.13727.pdf). 

## Using the code
1. Create a Nasdaq Data Link account to access the [free Quandl dataset](https://data.nasdaq.com/data/CHRIS-wiki-continuous-futures/documentation). This dataset provides continuous contracts for 600+ futures, built on top of raw data from CME, ICE, LIFFE etc.
2. Download the Quandl data with: `python -m data.download_quandl_data <<API_KEY>>`
3. Create Momentum Transformer input features with: `python -m examples.create_features_quandl`. In this example we use the 100 futures tickers which have i) the longest history ii) more than 90% of trading days have data iii) data up until at least Dec 2021.
4. Optionally, run the changepoint detection module: `python -m examples.concurent_cpd_quandl <<CPD_WINDOW_LENGTH>>`, for example `python -m examples.concurent_cpd_quandl 21` and `python -m examples.concurent_cpd_quandl 126`
5. Create Momentum Transformer input features, including CPD module features with: `python -m examples.create_features_quandl 21` after the changepoint detection module has completed.
6. To create a features file with multiple changepoint detection lookback windows: `python -m examples.create_features_quandl 126 21` after the 126 day LBW changepoint detection module has completed and a features file for the 21 day LBW exists.
7. Run one of the Momentum Transformer or Slow Momentum with Fast Reversion experiments with `python -m examples.run_dmn_experiment <<EXPERIMENT_NAME>>`

## Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture
> Deep learning architectures, specifically Deep Momentum Networks (DMNs) , have been found to be an effective approach to momentum and mean-reversion trading. However, some of the key challenges in recent years involve learning long-term dependencies, degradation of performance when considering returns net of transaction costs and adapting to new market regimes, notably during the SARS-CoV-2 crisis. Attention mechanisms, or Transformer-based architectures, are a solution to such challenges because they allow the network to focus on significant time steps in the past and longer-term patterns. We introduce the Momentum Transformer, an attention-based architecture which outperforms the benchmarks, and is inherently interpretable, providing us with greater insights into our deep learning trading strategy. Our model is an extension to the LSTM-based DMN, which directly outputs position sizing by optimising the network on a risk-adjusted performance metric, such as Sharpe ratio. We find an attention-LSTM hybrid Decoder-Only Temporal Fusion Transformer (TFT) style architecture is the best performing model. In terms of interpretability, we observe remarkable structure in the attention patterns, with significant peaks of importance at momentum turning points. The time series is thus segmented into regimes and the model tends to focus on previous time-steps in alike regimes. We find changepoint detection (CPD) , another technique for responding to regime change, can complement multi-headed attention, especially when we run CPD at multiple timescales. Through the addition of an interpretable variable selection network, we observe how CPD helps our model to move away from trading predominantly on daily returns data. We note that the model can intelligently switch between, and blend, classical strategies - basing its decision on patterns in the data.

## Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection
> Momentum strategies are an important part of alternative investments and are at the heart of commodity trading advisors (CTAs). These strategies have, however, been found to have difficulties adjusting to rapid changes in market conditions, such as during the 2020 market crash. In particular, immediately after momentum turning points, where a trend reverses from an uptrend (downtrend) to a downtrend (uptrend), time-series momentum (TSMOM) strategies are prone to making bad bets. To improve the response to regime change, we introduce a novel approach, where we insert an online changepoint detection (CPD) module into a Deep Momentum Network (DMN) pipeline, which uses an LSTM deep-learning architecture to simultaneously learn both trend estimation and position sizing. Furthermore, our model is able to optimise the way in which it balances 1) a slow momentum strategy which exploits persisting trends, but does not overreact to localised price moves, and 2) a fast mean-reversion strategy regime by quickly flipping its position, then swapping it back again to exploit localised price moves. Our CPD module outputs a changepoint location and severity score, allowing our model to learn to respond to varying degrees of disequilibrium, or smaller and more localised changepoints, in a data driven manner. Back-testing our model over the period 1995-2020, the addition of the CPD module leads to an improvement in Sharpe ratio of one-third. The module is especially beneficial in periods of significant nonstationarity, and in particular, over the most recent years tested (2015-2020) the performance boost is approximately two-thirds. This is interesting as traditional momentum strategies have been underperforming in this period.


## References
Please cite our papers with:
```bib
@article{wood2021trading,
  title={Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture},
  author={Wood, Kieran and Giegerich, Sven and Roberts, Stephen and Zohren, Stefan},
  journal={arXiv preprint arXiv:2112.08534},
  year={2021}
}

@article {Wood111,
	author = {Wood, Kieran and Roberts, Stephen and Zohren, Stefan},
	title = {Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection},
	volume = {4},
	number = {1},
	pages = {111--129},
	year = {2022},
	doi = {10.3905/jfds.2021.1.081},
	publisher = {Institutional Investor Journals Umbrella},
	issn = {2640-3943},
	URL = {https://jfds.pm-research.com/content/4/1/111},
	eprint = {https://jfds.pm-research.com/content/4/1/111.full.pdf},
	journal = {The Journal of Financial Data Science}
}
```

The Momentum Transformer uses a number of components from the Temporal Fusion Transformer (TFT). The code for the TFT can be found [here](https://github.com/google-research/google-research/tree/master/tft).

## Sample results
Will be made available soon. 

## Subsequent work
We also have a follow-up paper: [Few-Shot Learning Patterns in Financial Time-Series for Trend-Following Strategies](https://arxiv.org/abs/2310.10500)

# DPA (Deep Portfolio Allocation)

## 개요

이 프로젝트는 딥러닝을 활용한 포트폴리오 할당 전략을 구현한 코드베이스입니다. 주요 특징으로는 Elastic Weight Consolidation (EWC)을 통한 온라인 학습과 동적 lambda 조정 기능이 포함되어 있습니다.

## 주요 기능

### 1. Elastic Weight Consolidation (EWC)
- **정적 EWC**: 고정된 lambda 값(기본값: 1.0)을 사용한 지식 보존
- **동적 EWC**: 이전 윈도우 결과에 따라 lambda 값을 동적으로 조정

### 2. 동적 Lambda 조정 기능

온라인 학습 과정에서 이전 윈도우의 성능 지표(Sharpe ratio)에 따라 EWC lambda 값을 자동으로 조정합니다:

#### 조정 로직
- **성능 개선 시**: lambda 감소 → 더 적극적인 학습 허용
- **성능 저하 시**: lambda 증가 → 이전 지식 보존 강화
- **성능 유지 시**: lambda 유지 → 임계값 미만 변화는 무시

#### 주요 함수들
- `adjust_lambda_dynamically()`: 성능 변화에 따른 lambda 조정
- `calculate_adaptive_lambda()`: 휴리스틱 방법과 성능 기반 조정 결합
- `estimate_lambda_ewc()`: Fisher 정보 기반 휴리스틱 lambda 계산

#### 사용 방법
```python
# 동적 lambda 조정 활성화
params["lambda_ewc"] = "auto"

# 온라인 학습 실행
run_online_learning(
    experiment_name="your_experiment",
    features_file_path="your_data.csv",
    params=params,
    window_size=8064,
    delta=2016,
    fine_tune_epochs=1
)
```

#### 결과 분석
- `lambda_sharpe_history.csv`: 각 윈도우별 lambda와 Sharpe ratio 기록
- 실시간 lambda 조정 로그 출력
- 성능 변화에 따른 lambda 변화 추적

### 3. 온라인 학습
- 슬라이딩 윈도우 기반 연속 학습
- EWC를 통한 catastrophic forgetting 방지
- 실시간 성능 모니터링

## 설치 및 실행

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 동적 Lambda 조정 테스트
```bash
python examples/test_dynamic_lambda.py
```

### 온라인 학습 실행
```bash
python examples/infer_launcher.py
```

## 설정 옵션

### EWC 관련 설정
```python
params = {
    "lambda_ewc": "auto",  # "auto" 또는 고정값 (예: 1.0, 100.0)
    "lambda_ewc_target_ratio": 0.5,  # 휴리스틱 계산용 목표 비율
}
```

### 동적 조정 파라미터
```python
# adjust_lambda_dynamically 함수에서 조정 가능
adjustment_factor=0.2,      # 조정 강도 (0~1)
performance_threshold=0.1,  # 성능 변화 임계값
min_lambda=0.1,            # 최소 lambda 값
max_lambda=10.0,           # 최대 lambda 값
```

## 파일 구조

```
dpa/
├── mom_trans/
│   ├── new_inference.py      # 온라인 학습 및 동적 lambda 조정
│   ├── deep_momentum_network.py
│   └── momentum_transformer.py
├── examples/
│   ├── infer_launcher.py     # 온라인 학습 실행 예제
│   └── test_dynamic_lambda.py # 동적 lambda 조정 테스트
└── README.md
```

## 성능 모니터링

온라인 학습 중 다음과 같은 정보를 확인할 수 있습니다:

1. **실시간 로그**: 각 윈도우별 Sharpe ratio와 lambda 값
2. **동적 조정 로그**: lambda 변화 원인과 조정량
3. **히스토리 파일**: 전체 학습 과정의 lambda와 성능 추이

## 주의사항

- 동적 lambda 조정은 첫 번째 윈도우 이후부터 적용됩니다
- 성능 변화가 임계값보다 작으면 lambda 조정이 발생하지 않습니다
- lambda 값은 설정된 최소/최대 범위 내에서만 조정됩니다

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
