# Deep Feature Extraction for Long-Term Delinquency Transition Prediction under Short Observation Windows

[[English]](#English) [[Korean]](#Korean)

---

**English**<a name="English"></a>

This repository is designed to ensure the reproducibility of the article. <br />
**Note**: This study was conducted with KCB (Korea Credit Bureau) Data.

---

## Research Abstract

The early identification of borrowers at risk of transitioning into long-term delinquency is a critical task in credit risk management. However, in real-world financial environments, risk prediction is often constrained by short observation windows and aggregated tabular credit data, which limit the effectiveness of conventional time-series-based or handcrafted feature approaches. To address this issue, this study proposes a deep feature extraction framework that leverages representation learning as a complementary component to traditional classifiers, rather than as a standalone predictor. The proposed approach integrated transformer-based global interaction learning with TabNet-inspired soft feature selection to generate risk-preserving latent representations from low-information credit data. Using a real-world multidebtor credit dataset, extensive experiments demonstrated that the extracted representations consistently improve long-term delinquency transition predictions, particularly when combined with tree-based ensemble models under severe class imbalance. Beyond predictive performance, distance-based and explainability analyses show that the learned feature space enhances class separability and provides interpretable insights into delinquency transition risks. These results suggest that deep representation learning is a practical and explainable solution for early delinquency transition detection in realistic financial tabular data settings, where temporal information is limited.

**Keywords**: Delinquency Transition; Deep Learning Feature Extraction; Credit Risk Prediction; Short Observation Window; Hybrid Learning Framework; Card Advance


---

## Significance of Cash Advance Usage in Delinquency Transition Risk

| ![image](https://github.com/user-attachments/assets/622718f9-c0ca-44b6-85e5-6df7d073269e) | ![image](https://github.com/user-attachments/assets/760fb7f6-4260-4260-9404-75c718360295) |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |

<p align='center'>Figure 1. Proportion of Cash Advance Users Among Delinquent Borrowers (Left) and Delinquency Rates by Cash Advance Usage (Right)</p>

**Note**: Analyzing our dataset, as shown in Figure 1, among the 5,201 currently delinquent borrowers, 30.74% (1,599 individuals) held a CA. Comparing CA holders and non-holders across the entire population, non-CA holders exhibited lower delinquency risk (2.64% vs. 5.71%).


---

## Method of Our Research

<img width="1741" height="1034" alt="KCB_Figure-002" src="https://github.com/user-attachments/assets/9beb5e66-744f-4cd8-be7c-1afc49f8e8ba" />
<p align='center'>Figure 2. Configuration Diagram of Feature Extract</p>

Deep-learning-based feature extraction aims to capture latent representations from existing financial variables while simultaneously reflecting the relative importance of each variable. To this end, as illustrated in Figure 2, we leverage the Transformer and TabNet model architectures, which are commonly used for feature extraction, to progressively extract representation information from complementary perspectives. <br />

<br />

<img width="2500" height="1080" alt="001" src="https://github.com/user-attachments/assets/fbea8f78-90f2-4ce8-99dd-e652b20db7d6" />
<p align='center'>Figure 3. Overall of Deep Learning-based Model Architecture</p>

To facilitate a more effective feature extraction from the data, we designed a hybrid encoder architecture that combines the global interaction learning capability of the Transformer-based self-attention block with the soft feature selection block of TabNet. The encoder is not designed as a standalone deep learning classifier but rather serves as a feature generation module that enhances the representational capacity of the input feature space, reshaping it into a risk-separable representation using the deep learning components described earlier for feature generation. The overall architecture of the model is illustrated in Figure 3. </br>

---

### Experiment Code and Datasets

#### Experiment code

- [Data Preprocess](Data_Preprocessing.ipynb)
- [Experiments](final_experiment.ipynb)

</br >

#### Dataset

- [Original Datasets](Datasets/Is_longterm.csv)

The original dataset was postprocessed to fit the study cohort.

- [Feature Extractor_Train](Datasets/X_train_mega_39.csv)
- [Feature Extractor_Test](Datasets/X_test_mega_39.csv)

This is the dataset produced by the feature extractor, which corresponds to Figure 2.

</br >

### Experimental Setup

All experiments in this study were conducted under a consistent environment and evaluation criteria. The computational environment comprised a server equipped with 128 GB of RAM, an Intel® Xeon® w5-2455X CPU, and an NVIDIA GeForce RTX 4090 GPU with 24 GB of memory. Models were implemented using Python 3.13.5, with numpy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1, and PyTorch 2.8.0.

---

If you have any questions regarding the research, please contact us at the email below. </br>

<a href=mailto:chungyn@hanyang.ac.kr> <img src="https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white&link=mailto:chungyn@hanyang.ac.kr"> </a>

chungyn@hanyang.ac.kr </br>

---

---

**Korean**<a name="Korean"></a>

해당 레포지토리는 논문의 재현성을 위해 구현되었습니다. <br />
**참고**: 본 연구는 한국신용정보회사(KCB)의 데이터를 이용하여 진행되었습니다.

---

## Research Abstract

다중채무자가 장기 연체로 전이되는 위험을 조기에 식별하는 것은 효과적인 신용 리스크 관리에 있어 매우 중요하지만, 짧은 관측 구간과 집계형 금융 데이터만이 제공되는 금융 환경에서는 이를 정확히 예측하는 데 한계가 존재합니다. 따라서 본 연구는 이러한 제약을 보완하기 위해 딥러닝 모델을 최종 예측기가 아닌 특징 추출기로 활용합니다. Transformer 기반 전역적 상호작용 학습과 TabNet에서 착안한 Soft Feature Selection을 결합한 프레임워크는 기존 머신러닝 모델 및 단일 딥러닝 접근 대비 일관되게 소폭 상승한 예측 성능을 보였습니다. 특히 심한 클래스 불균형과 제한된 시간 정보 환경에서도 장기 연체로의 전이 신호를 조기에 포착하는 데 효과적임을 확인하며, 추출된 특징들에 대한 설명 가능성을 분석합니다. 이를 통해 딥러닝 기반 특징 추출과 트리 기반 분류기를 체계적으로 결합하는 접근이 실제 금융 tabular 데이터 환경에서 장기 연체 전이 예측을 위한 실용적이고 효과적인 해결책이 될 수 있음을 시사합니다.

**Keywords**: Delinquency Transition; Deep Learning Feature Extraction; Credit Risk Prediction; Short Observation Window; Hybrid Learning Framework; Card Advance

---

## Significance of Cash Advance Usage in Delinquency Transition Risk

| ![image](https://github.com/user-attachments/assets/622718f9-c0ca-44b6-85e5-6df7d073269e) | ![image](https://github.com/user-attachments/assets/760fb7f6-4260-4260-9404-75c718360295) |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |

<p align='center'>그림 1. 연체 차입자 중 현금 선지급 이용자 비율(왼쪽) 및 현금 선지급 이용 여부에 따른 연체율(오른쪽)</p>

**Note**: 그림 1에서 볼 수 있듯이, 현재 연체 중인 5,201명의 차용자 중 30.74%(1,599명)가 신용조회(CA)를 보유하고 있는 것으로 나타났습니다. 전체 차용자 집단을 대상으로 신용조회 보유자와 비보유자를 비교했을 때, 비보유자의 연체 위험이 더 낮았습니다(2.64% vs 5.71%).

---

## Method of Our Research

<img width="1741" height="1034" alt="KCB_Figure-002" src="https://github.com/user-attachments/assets/9beb5e66-744f-4cd8-be7c-1afc49f8e8ba" />
<p align='center'>그림 2. 특징 추출기 구성도</p>

딥러닝 기반 특징 추출은 기존 금융 변수에서 잠재적 표현을 포착하는 동시에 각 변수의 상대적 중요도를 반영하는 것을 목표로 합니다. 이를 위해 그림 2에서 보여주는 것처럼 특징 추출에 일반적으로 사용되는 Transformer 및 TabNet 모델 아키텍처를 활용하여 상호 보완적인 관점에서 표현 정보를 점진적으로 추출합니다. <br />

<br />

<img width="2500" height="1080" alt="001" src="https://github.com/user-attachments/assets/fbea8f78-90f2-4ce8-99dd-e652b20db7d6" />
<p align='center'>Figure 3. 딥러닝 기반 모델 아키텍처의 개요도</p>

데이터로부터 보다 효과적인 특징 추출을 위해, 본 연구에서는 Transformer 기반 셀프 어텐션 블록의 전역 상호작용 학습 능력과 TabNet의 소프트 특징 선택 블록을 결합한 하이브리드 인코더 아키텍처를 설계했습니다. 이 인코더는 독립적인 딥러닝 분류기로 설계된 것이 아니라, 입력 특징 공간의 표현력을 향상시키고, 앞서 설명한 특징 생성 구성 요소를 활용하여 위험 분리 가능한 표현으로 재구성하는 특징 생성 모듈 역할을 합니다. 모델의 전체 아키텍처는 그림 3에 나타나 있습니다. </br>

---

### Experiment Code and Datasets

#### Experiment code

- [Data Preprocess](Data_Preprocessing.ipynb)
- [Experiments](final_experiment.ipynb)

</br >

#### Dataset

- [Original Datasets](Datasets/Is_longterm.csv)

원본 데이터 세트는 연구 코호트에 맞게 후처리되었습니다.

- [Feature Extractor_Train](Datasets/X_train_mega_39.csv)
- [Feature Extractor_Test](Datasets/X_test_mega_39.csv)

해당 데이터셋은 그림 2와 같이 특징 추출기가 생성한 데이터셋을 csv로 추출한 값입니다.

</br >

### Experimental Setup

본 연구의 모든 실험은 일관된 환경과 평가 기준 하에서 수행되었습니다. 계산 환경은 128GB RAM, Intel® Xeon® w5-2455X CPU, 24GB 메모리를 갖춘 NVIDIA GeForce RTX 4090 GPU가 장착된 서버로 구성되었습니다. 모델 구현에는 Python 3.13.5, numpy 2.2.6, pandas 2.3.2, scikit-learn 1.7.1, PyTorch 2.8.0이 사용되었습니다.

---

연구와 관련해서 질문이 있으시다면 아래 메일로 연락주세요.

<a href=mailto:chungyn@hanyang.ac.kr> <img src="https://img.shields.io/badge/Gmail-EA4335?style=flat-square&logo=Gmail&logoColor=white&link=mailto:chungyn@hanyang.ac.kr"> </a>

chungyn@hanyang.ac.kr </br>
