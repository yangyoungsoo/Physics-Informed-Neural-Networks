# Research: MLP 기반 PINNs 설계

## 목적

MLP 기반의 Physics-Informed Neural Network(PINNs) 구현.
다양한 기능을 모듈화하여 on/off 가능한 유연한 구조를 목표로 한다.

---

## 방법 / 기능 명세

### 1. 유연한 입출력 차원
- 입력 차원 자유 설정 (1D / 2D / 3D 등 사용자 정의)
- 출력 차원 자유 설정

### 2. 레이어 구조 사용자 정의
- 레이어 크기를 리스트로 입력하면 자동으로 모델 선언
- 예시: `[3, 64, 64, 64, 64, 64, 1]` → 입력 3, 은닉층 5개(64), 출력 1

### 3. Activation Function 선택
- 사용자가 활성화 함수를 선택 가능
- 후보: `tanh`, `sin`, `swish`, `relu`, `gelu` 등

### 4. 입력 인코딩 (On/Off)
| 기능 | 설명 |
|------|------|
| Fourier Feature Embedding (RFF) | 입력 좌표를 `[sin(Bx), cos(Bx)]`로 인코딩. 고주파 패턴 학습 향상. B는 랜덤 or 학습 가능 |
| Positional Encoding | 입력 좌표를 다중 주파수 `sin/cos`로 인코딩 (NeRF 스타일). `[sin(2^k πx), cos(2^k πx)]` 형태 |

### 5. On/Off 가능한 기능
| 기능 | 설명 |
|------|------|
| Residual Connection | 같은 크기 레이어 간 잔차 연결 |
| Skip Connection | 입력층 → 중간층으로 직접 연결 |
| NTK Parameterization (init) | 초기화 시 weight를 `1/sqrt(fan_in)`으로 스케일링 |
| NTK Adaptive Loss Weighting | 학습 중 각 loss term의 gradient norm을 주기적으로 계산해 `w_bc`, `w_ic`를 자동 조정 (Wang et al. 2022). `ntk_adaptive=True`, `ntk_update_every=N`으로 제어 |

### 6. `fit()` 함수
- 모델 클래스 내부에 `fit()` 메서드 포함
- 주요 인자:

| 인자 | 설명 |
|------|------|
| `pde_fn` | `(model, x) → residual` 형태의 PDE residual 함수 |
| `x_pde` | 내부 collocation points |
| `x_bc`, `u_bc` | 경계조건 점 및 타겟값 |
| `x_ic`, `u_ic` | 초기 변위 조건 (1차 시간 PDE 또는 2차 공통) |
| `x_ic_ut`, `ut_ic` | **초기 속도 조건** — 2차 시간 PDE (파동 방정식 등)에서 `∂u/∂t(x,0)` 지정. autograd로 계산 후 타겟과 MSE |
| `w_ic_ut` | 초기 속도 loss 가중치 |

> **2차 시간 PDE 처리 방법**  
> Wave equation처럼 `∂²u/∂t²`이 포함된 문제는 IC가 두 개 필요:  
> 1. `u(x, 0) = f(x)` → `x_ic`, `u_ic`로 처리  
> 2. `∂u/∂t(x, 0) = g(x)` → `x_ic_ut`, `ut_ic`로 처리 (autograd로 `du/dt` 계산)

### 7. Loss 출력
- `fit()` 실행 중 매 epoch(또는 n step)마다 loss 출력
- 출력 항목 예시:
  - `Total Loss`
  - `PDE Residual Loss`
  - `Boundary Condition Loss`
  - `Initial Condition Loss` (변위)
  - `IC Velocity Loss` (속도 — 2차 시간 PDE의 경우)

### 8. Loss History 저장
- `fit()` 실행 중 각 loss 항목을 dict 형태로 누적 저장
- 학습 완료 후 `model.loss_history`로 접근 가능
- 예시:
  ```python
  model.loss_history = {
      "total": [...],
      "pde": [...],
      "bc": [...],
      "ic": [...]
  }
  ```

### 9. Best Model 저장
- `fit()` 중 `total loss`가 갱신될 때마다 모델 가중치 자동 저장
- 저장 경로를 `fit()` 인자로 지정 가능 (기본값: `./best_model.pt`)
- 학습 완료 후 best weight를 자동으로 불러옴 (restore best weights)

---

## 결과

- 추후 실험 결과 기록 예정

---

## 참고문헌

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*.
- Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*.
