# OCC for AD
- Mean-Shifted Contrastive Loss for Anomaly Detection(AAAI 2023) 논문을 베이스로 개발되었습니다.
- 학습 코드: python main.py --dataset='plant' --label=0     (--dataset은 불러올 데이터셋 이름, --label 은 데이터셋 폴더에서 어떤 클래스를 정상 클래스로 둘 지 결정)
- 추론 코드: single_predictor.py 실행 ( 안에서 파라미터를 조정하여 추론 가능 )


# 설명
- 이 코드는 One Class Classification for Anomaly Detection으로 하나의 정상 클래스 외의 나머지 데이터를 이상 클래스로 분류한다. 
- 이 코드에서는 Plant 데이터셋(식물 데이터셋)을 제외한 나머지 데이터를 이상 데이터로 분류한다. 