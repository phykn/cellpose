# Cellpose 저장소 심층 감사

2026년 10월 13일 기준 감사. 기준 커밋은 `66507d4`이며, `src`의 27개 Python
모듈, 두 CLI, 설정, 의존성, 테스트, fixture 생성 스크립트와 문서를 검토했다.
학습·체크포인트, 데이터·전처리, 모델·추론을 병렬로 조사하고 담당자가 작성하지
않은 변경을 교차 검토했다. 확인된 보안 의존성 문제와 수치·입력 검증 결함을
수정했다. 큰 객체의 flow target 계산은 여전히 주요 CPU 비용이다.

## 심각도순 발견과 변경

심각도는 이 저장소에서 발생 가능한 영향과 입력 경로를 기준으로 한다.
잘못된 저수준 API 입력에서만 발생하는 문제는 정상 CLI 경로의 결함과 구분했다.
파일의 줄 번호는 이번 수정본 기준이다.

| 심각도 | 발생 조건과 영향 | 수정 근거 | 회귀 근거 |
|---|---|---|---|
| 높음 P1 | `torch` 버전 제약이 없어 악성 checkpoint를 안전하게 역직렬화하지 못하는 구버전도 설치 요건을 충족했다. 현재 설치된 2.14.0은 해당 취약점의 영향 범위 밖이다. | [requirements.txt:8](../requirements.txt#L8)에 `torch>=2.10.0`을 지정하고 README에 이유를 명시했다. | [test_security.py:31](../tests/test_security.py#L31): plain weights, training checkpoint, backbone의 사용자 정의 pickle 실행 차단. CVE exploit을 실행한 검증은 아니다. |
| 높음 P1 | NaN/Inf loss 또는 비유한 gradient가 발생해도 optimizer가 실행되어 모델과 저장 checkpoint가 오염됐다. 실제로 `loss=nan/inf`를 기록하고 checkpoint를 저장했다. | [trainer.py:98](../src/train/trainer.py#L98)에서 backward 전 loss를 검사하고, [trainer.py:103](../src/train/trainer.py#L103)에서 non-AMP optimizer 업데이트 전 gradient를 검사한다. | [test_trainer.py:168](../tests/train/test_trainer.py#L168), [test_trainer.py:186](../tests/train/test_trainer.py#L186): 3개 실패 재현 후 통과. 가중치·optimizer·step·저장 상태 보존 확인. |
| 중간 P2 | 직접 호출한 instance 분류 API가 소수 mask ID `1.5`를 `1`로 절삭하고 음수·3차원 마스크도 처리했다. 극단 int64 logits는 뺄셈 overflow로 작은 logit의 클래스를 score 약 0.731로 선택했고, complex logits는 허수를 버렸다. | [classify.py:11](../src/predict/classify.py#L11)의 공통 mask 검증과 [classify.py:12](../src/predict/classify.py#L12)의 실수 부동소수점 logits 검증. 정상 추론이 전달하는 float32 계약을 유지한다. | [test_classification.py:28](../tests/test_classification.py#L28), [test_classification.py:40](../tests/test_classification.py#L40): 잘못된 mask 3종과 logits 2종 실패 재현 후 통과. |
| 낮음 P3 | 직접 dynamics에 complex flow/logit을 전달하면 허수가 조용히 유실됐다. 유한 float64 `1e300`은 float32 변환에서 overflow되어 NaN 위치와 `IndexError`를 만들었다. | [dynamics.py:190](../src/predict/dynamics.py#L190), [dynamics.py:226](../src/predict/dynamics.py#L226): 실수 dtype과 float32 변환 범위를 사전 검증한다. 정상 float32에는 추가 범위 비교를 하지 않는다. | [test_dynamics.py:75](../tests/test_dynamics.py#L75), [test_dynamics.py:84](../tests/test_dynamics.py#L84): 3개 실패 재현 후 통과. |
| 낮음 P3 | 공간 차원이 0인 mask를 NumPy/TIFF로 저장하지만 다시 읽을 수 없었다. PNG는 출력 디렉터리를 만든 뒤 하위 라이브러리에서 실패했다. | [image.py:50](../src/data/image.py#L50): 공통 `check_masks`를 파일·디렉터리 생성 전에 호출한다. | [test_image.py:73](../tests/test_image.py#L73): `(0,2)`·`(2,0)` × NumPy·PNG·TIFF, 6개 실패 재현 후 통과. |

PyTorch의 공식 공지는 `weights_only=True`를 우회하는
[CVE-2026-24747](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)의
영향 범위를 2.9.1 이하, 수정 버전을 2.10.0 이상으로 명시한다. 새 하한은
2.5.1 이하에 영향을 주는
[CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)도
제외한다. 이는 의존성 설치 요건의 보강이며, 현재 환경에서 임의 코드 실행을
발견했다는 뜻은 아니다.

## 모듈 구조와 의존성

학습은 `run_train → config/build → dataset/model/trainer`, 추론은
`run_predict → config/build → predict`로 진행된다. `prepare`의 순수 배열·tensor
계산을 학습과 추론이 공유한다. 내부 순환 import는 발견하지 못했다.
CLI·파일 형식·모델 tensor 이름·정상 입력의 계산 규칙은 유지했다.

| 모듈 | 책임과 주요 의존성 | 검토 결과 |
|---|---|---|
| `run_train.py` | 설정 우선순위·seed·새 run 디렉터리·학습 시작 | 기존 run 덮어쓰기 거절, train/resume CLI 통합 검사 통과 |
| `run_predict.py` | sidecar 설정 탐색·추론·mask 및 labels JSON 출력 | 저장 설정을 이용하는 segmentation/classification 통합 검사 통과 |
| `src/__init__.py` | `FLOW_SCALE=5`, `MODEL_STRIDE=8` 공통 상수 | target·loss·inference의 축과 scale 연결 확인 |
| `src/config.py` | PyYAML 설정 병합·경로 해석·장치 및 수치 검증 | `safe_load`, CLI 우선순위, YAML project-root 상대 경로 검사 |
| `src/build.py` | 모델·Dataset·DataLoader·AdamW·Trainer 조립 | 모델 설정 선검증, resume/weights 배타성, loader 설정 검사 |
| `src/checkpoint.py` | plain tensor weights·metadata 검증·sidecar 탐색 | `weights_only=True`, strict loading, classifier 호환성 및 원자적 파일 교체 검사 |
| `src/model/config.py` | backbone·stride·class vocabulary·metadata 계약 | 클래스 순서와 암묵적 기본값의 호환성 검사 |
| `src/model/dinov3.py` | 고정 DINOv3 API 호출·로컬 backbone loading | 허용 backbone 제한, 다운로드 없는 `pretrained=False`, strict local loading 확인 |
| `src/model/cellpose.py` | patch projection 조정·동결·dense heads·pixel shuffle | 출력 해상도·segmentation/classification gradients 및 실제 DINO smoke 검사 |
| `src/data/image.py` | NumPy/Pillow/tifffile IO·stem pairing | pickle 거절, sparse/uint64 ID 보존, PNG 범위 검사, 빈 출력 수정 |
| `src/data/dataset.py` | 읽기·정규화·paired crop·flow/class target 조립 | image/mask 공간 일치, annotation 완전성, crop 후 원래 ID 유지 검사 |
| `src/data/augment.py` | Torch RNG를 쓰는 paired crop·90도 회전·반사 | image/mask alignment, padding·view를 tensor로 바꾸는 경계 확인 |
| `src/data/label.py` | JSON ID→class 파싱·pixel class target | 중복 키·비정규 ID·누락/추가 ID 거절, compact inverse mapping 확인 |
| `src/prepare/convert.py` | 실수 영상의 채널 배치·percentile 정규화 | int64/uint64 offset·float64 차이 보존, finite float32 결과 검사 |
| `src/prepare/mask.py` | 2D 정수 mask 검증·compact 재번호 | 읽기·저장·분류의 공통 불변식, 큰 sparse ID가 배열 크기를 늘리지 않음 |
| `src/prepare/resize.py` | stride 정렬·padding offset | 홀수 영상 크기 복원과 tiled 테스트 확인 |
| `src/prepare/flow.py` | NumPy/SciPy diffusion·중심 선택·unit flow target | 고정 upstream fixture 및 round trip 검사, CPU 비용 측정 |
| `src/train/loss.py` | flow MSE/2·foreground BCE·선택적 class CE | 채널 순서·scale·빈 crop의 0 class gradient 유지 |
| `src/train/trainer.py` | 배치 반복·AMP·optimizer·metrics·저장 | 비유한 학습 중단 수정, 정상 수동 업데이트와 동일성 검사 |
| `src/train/checkpoint.py` | optimizer/scaler/config/step/RNG 저장·복원 | legacy 및 v2 복원, 다음 optimizer update, 저장 실패 보존 검사 |
| `src/predict/inference.py` | normalize·eval/inference mode·CPU 결과 조립 | 성공/실패 후 각 하위 모듈의 training flag 복원 확인 |
| `src/predict/tile.py` | stride 정렬 타일·cosine blend·crop 복원 | float32 누적, 겹침 overflow 방지, 입력/parameter gradients 검사 |
| `src/predict/dynamics.py` | Euler integration·endpoint clustering·flow QC·hole/size filter | dtype/range 수정, 중첩 객체 보존, sparse labels·upstream 검사 |
| `src/predict/classify.py` | softmax 확률의 instance 평균·ID→class 결과 | mask/logits 검증 수정, sparse/uint64 ID 및 분류 대응 검사 |
| `src/{data,model,prepare,predict,train}/__init__.py` | 5개 package 경계 | 추가 실행 로직 없음 |
| `scripts/generate_flow_fixture.py` | 선택한 upstream Python 함수로 offline fixture 생성 | 전달한 Python source를 실행하는 개발 도구. 신뢰한 고정 upstream source를 전제로 하며 일반 image/weights 경로에서 호출되지 않음 |

책임 분리가 명확하고 기존 테스트가 주요 데이터 계약을 다루므로 구조 이동은
필요하지 않았다. 새 검증은 공통 mask 소유자와 실제 변환·optimizer 경계에 추가했다.
외부 API·클래스·설정 옵션·런타임 의존성은 추가하지 않았다.

## 의존성과 보안 검증

실행 환경은 Windows, Python 3.13.15, CPU PyTorch 2.14.0이다. 직접 의존성은
DINOv3, NumPy 2.5.3, Pillow 12.3.0, PyYAML 6.0.3, SciPy 1.18.1,
tifffile 2026.9.20, PyTorch이며 개발 도구는 pytest 9.1.1, Ruff 0.16.9였다.
DINOv3는 torchvision·torchmetrics·scikit-learn·OmegaConf 등을 전이 의존성으로
설치한다. 설치 metadata의 URL이 `requirements.txt`의 커밋
`6876159a11b4df116f30f667f8c9888617df0751`과 일치했다.

`pip check`는 충돌을 발견하지 않았다. `pip-audit --path .venv/Lib/site-packages`
결과는 설치 배포판 40개 중 PyPI 조회 가능한 39개에서 알려진 취약점 0건이었다.
DINOv3는 PyPI 감사에서 제외되어 고정 출처와 실제 API 실행을 별도로 확인했다.
독립적인 PyPI 버전별 JSON 조회도 같은 39개에서 알려진 취약점 0건이었다.
이 결과는 검사 날짜의 설치 환경과 알려진 advisory에 한정되며 미래 버전이나
DINOv3 전체 소스의 무결함을 보장하지 않는다. DINOv3 외 패키지는 전체 버전 잠금이
없으므로 다른 날짜의 설치 조합은 달라질 수 있다.

[test_security.py](../tests/test_security.py)는 세 Torch loader에서 임의 pickle
함수의 파일 쓰기가 실행되지 않는지 확인하고, NumPy object array와 Python object
YAML의 역직렬화를 거절하는지도 검사한다. 저장 구현은 이미 임시 파일 저장 후
`os.replace`를 사용하므로 변경하지 않았다.
[test_checkpoint.py:91](../tests/train/test_checkpoint.py#L91)의 두 fault injection
테스트가 부분 저장·디스크 오류 이후 기존 파일 바이트와 복원값의 보존을 확인한다.

## 성능 병목과 비교 실험

확인된 주요 비용은 [flow.py:46](../src/prepare/flow.py#L46)의 diffusion이다.
foreground 픽셀 수를 F, 반복 수를 S라 하면 시간은 O(S×F), 임시 메모리는
O(HW+9F)이다. 기본 S는 가장 큰 객체의 bounding-box extent에 비례한다.
Dataset은 crop/augmentation 이후 target을 다시 계산하므로 큰 객체가 포함된
crop에서 CPU 비용이 커진다.

384×384 영상, 동일한 foreground 65,536픽셀, 시간 3회 및 별도 tracemalloc
1회 측정 결과다. 정규화 입력은 seed 7의 float32 정규분포 배열이다.
큰 객체는 `[64:320,64:320]`, 작은 객체는 간격 12의 32×32 grid에 8×8씩 배치했다.

| 연산 | 큰 256×256 객체 1개 중앙 시간 | 작은 8×8 객체 1024개 중앙 시간 | 큰 경우 peak 임시 메모리 |
|---|---:|---:|---:|
| `normalize_image` | 0.0058초 | 동일 영상 | 5.06 MiB |
| `flow_targets` | 90.2784초 | 4.3225초 | 23.71 MiB |
| `class_target` | 0.0484초 | 0.1808초 | 4.64 MiB |

호스트에서 다른 작업도 실행 중이었고 큰 flow 측정은 63.6510~141.7362초로
변동했다. 이는 병목 위치를 확인하는 관측값이며 전용 환경의 처리량이나 정확한
배수 비교로 사용할 수 없다.

flat neighbor index와 `np.take`/multiply/sum의 재사용 buffer를 적용한 대안도
검토했다. 16개 mask·반복 조건과 upstream fixture에서 bitwise 동일했지만,
128×128 영상의 96×96 객체에 대해 실행 순서를 번갈아 세 쌍 측정했을 때
원본 중앙 1.4578초·peak 3.2404 MiB, 후보 1.8858초·2.6802 MiB였다.
peak 메모리는 17.3% 감소했으나 세 쌍 모두 느려 CPU 시간은 중앙 29.4% 증가했다.
따라서 후보는 적용하지 않았다. 반복 수나 정밀도를 낮추는 변경은 학습 target과
재구성 결과를 바꾸므로 이번 수정에 포함하지 않았다.

다음은 코드 경로상 남은 비용이며 이 감사에서 실사용 GPU 병목으로 측정한 것은 아니다.

| 경로 | 비용과 영향 |
|---|---|
| [tile.py:33](../src/predict/tile.py#L33) | 타일은 모델의 한 번 입력 크기를 제한하지만 누적 output/weight는 전체 영상에 할당한다. K classes이면 output channel은 3+K이며 메모리는 영상 면적과 클래스 수에 비례한다. |
| [dynamics.py:36](../src/predict/dynamics.py#L36) | `niter × foreground` bilinear gather 연산. 추론 output은 CPU로 이동한 뒤 flow tracking 장치로 다시 옮겨지고 QC는 NumPy/SciPy에서 수행된다. |
| [trainer.py:75](../src/train/trainer.py#L75) | 매 step metrics 파일 열기·쓰기, 주기적 weights와 optimizer 상태 저장. 실제 저장장치와 모델 크기에 따른 측정이 필요하다. |

## 검증 결과와 재현

수정 전 전체 pytest는 **142 passed**였다. 새 테스트로 학습 오류 3건, 분류 입력
오류 5건, dynamics 오류 3건, 빈 mask 저장 6건을 수정 전 재현했다.
기능 수정 1차 전체 실행은 **164 passed**였고, 이후 CPU AMP 회복 테스트 및
분류 logits 테스트를 추가했다. 최종 수정본은 **167 passed in 45.80s**였다.
기존 142개에 25개 검증 사례를 추가했으며 실패나 skip은 없었다.

| 최종 검사 | 결과 |
|---|---|
| 전체 pytest | 167 passed, 45.80초 |
| Ruff lint | 통과 |
| Ruff format check | 49 files already formatted |
| pip check | No broken requirements found |
| pip-audit | 39개 조회, 알려진 취약점 0건, DINOv3 1개 제외 |

독립 교차 검증에서는 보안·이미지·추론 관련 75개 테스트가 통과했다.
[test_trainer.py:201](../tests/train/test_trainer.py#L201)은 실제 CPU GradScaler로
AMP 분기의 Inf gradient 업데이트 생략, scale 128→64 감소, 다음 정상 step의
학습 회복을 확인한다. CUDA 실행 검증을 대체하는 주장은 하지 않는다.

실제 설치 DINOv3 `vits16`을 다운로드 없이 random initialization하여
stride 6·8·16에서 48×48 입력의 forward/backward, segmentation·classification head의 gradient,
patch projection 동결, 35×41 홀수 입력의 tiled 출력 크기·유한성을 확인했다.
기존 합성 train/resume/predict 테스트는 segmentation과 classification을 모두 다룬다.

Windows PowerShell에서 재현할 기본 검사:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest -q --durations=10
.\.venv\Scripts\python.exe -m ruff check src tests scripts run_train.py run_predict.py
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts run_train.py run_predict.py
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\uv.exe tool run --isolated pip-audit --path .venv/Lib/site-packages
```

검사 중 발생한 오류도 원인을 구분했다. 병렬 작업의 중간 실행에서 새 분류
테스트 3개가 구현 반영 전에 실패했으며 이후 분류 전체·교차 검증에서 통과했다.
추가된 image 테스트의 CRLF/LF 혼합으로 포맷 검사가 한 차례 실패하여 기존
형식으로 정리하고 재검사했다. 의존성 감사 도구의 시작 지연은 별도 cache 실행과
독립 PyPI 조회로 검증을 이어갔으며 최종 `pip-audit`도 정상 종료했다.

## 검증 범위의 한계

현재 환경에 CUDA가 없어 실제 CUDA AMP·CUDA RNG 복원·GPU peak memory와
처리량은 재검증하지 않았다. 다른 대형 ViT backbone, gated pretrained weights의
실제 세포 분할·분류 품질, 실데이터 전체 학습은 이번 결과에 포함되지 않는다.
resume은 모델·optimizer·scaler·프로세스 RNG를 복원하지만 DataLoader의 sampler
위치와 worker/prefetch 상태는 저장하지 않는다. 기존 계약대로 새 data pass를
시작하며 다음 batch 순서의 bitwise 재현을 보장하지 않는다.
