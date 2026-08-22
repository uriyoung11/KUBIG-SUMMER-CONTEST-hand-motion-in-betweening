## v1. SILK 베이스라인
- 공식 코드 미존재
#### 논문 대비 우리가 그대로 따른 것
- Transformer 인코더 **6층, 8-head**, `d_model=1024`, `d_ff=4096`, **Pre-LN**
- **AdamW + Noam 학습률 스케줄러**, batch size **64**
- 입력 구조: **C개 컨텍스트 프레임 + M개 빈(0) 프레임 + 목표 키프레임 1개**
- 빈 프레임은 **0으로 채우고 attention masking 없음** (모든 프레임이 서로 attend)
- **단일 L1 손실**
- 학습 시 M(가림 길이)을 5~30 사이에서 균일 샘플링, 평가는 5/15/30/45 고정
- 데이터 슬라이스 오프셋 5 (논문 대비 4배 촘촘한 샘플링)

#### 손 도메인에 맞게 조정한 것 (논문과 다른 지점, 명시)
- **회전(quaternion/6D) 특징 전부 제외** — 우리 데이터(SHREC, H2O, InterHand2.6M 등)는
  3D 관절 좌표만 제공하고 회전 정보가 없음. 원 논문의 `d_in=18J+8`, `d_out=9J+4` 대신
  **위치+속도만 사용**: `d_in = 6J`(위치 3J + 선속도 3J), `d_out = 3J`(위치만).
- **"루트를 지면에 투영"하는 개념 없음** — 몸은 골반을 지면에 투영해 root를 만들지만,
  손에는 이런 개념이 없어 **손목을 root 삼아 상대좌표로 정규화**(기존 파이프라인과 동일한 방식).
- **Relative positional encoding의 정확한 구현 방식은 논문에 상세 공개 안 됨** — 학습 가능한
  절대 위치 임베딩(learned absolute positional embedding)으로 근사 구현. (논문은 [25]의 방식을
  따른다고만 언급하고 구체적 수식은 미공개)


https://github.com/user-attachments/assets/17d93026-dcd2-470c-8df9-f7a672673f13


## v2. v1 + 뼈 길이 안정화

Bone-direction 분해 + Forward Kinematics

**문제**: 위치(XYZ)를 관절마다 독립적으로 회귀하면, 뼈 길이(관절 간 거리)가 프레임마다
미세하게 흔들리는 현상이 실제로 관찰됨(3D 시각화로 확인).

**문헌 조사 결과**:
- Skeleton Consistency Loss(뼈 길이/방향 불일치에 페널티를 주는 손실항, Zanfir et al.,
  arXiv:2109.10257)를 시도해볼 수 있으나,
- "Anatomy-aware 3D Human Pose Estimation with Bone-based Pose Decomposition"
  (arXiv:2002.10322)은 이런 손실 기반 접근을 직접 베이스라인으로 테스트한 결과
  **뼈 길이 일관성을 손실 함수만으로 강제하는 건 거의 효과가 없었다**고 명시
- **위치를 직접 예측하지 않고 "뼈 방향 + 뼈 길이"로 분해해서 예측**한 뒤
  forward kinematics로 위치를 복원하는 구조적 접근이 훨씬 효과적이었음.

**우리 상황에 유리한 점**: 
- 일반적인 전신 자세 추정은 뼈 길이 자체를 모르는 상태(사진
한 장)에서 추정해야 하지만, 우리는 같은 시퀀스의 **컨텍스트 프레임(실제 관측된 값)에서
진짜 뼈 길이를 이미 알고 있음**
- 짧은 시간 동안 손 뼈 길이는 물리적으로 불변이므로,
**뼈 길이는 컨텍스트에서 계산한 고정값을 쓰고, 모델은 뼈 "방향"(unit vector)만 예측**하면
뼈 길이 불일치가 구조적으로 원천 차단됨
- 또한 손목 기준 정규화를 이미 쓰고 있어 손목(root)은
항상 원점이므로 별도 예측이 필요 없음

https://github.com/user-attachments/assets/5836b37b-bc5e-44a3-af3a-b501bb564526


## v3. Diffusion SILK 1 (diffusion_silk_notebook.ipynb)

**SILK 구조 유지**

- **6층·8헤드 Transformer 인코더**(`d_model=1024`, `d_ff=4096`, Pre-LN)를 그대로 가져다 씀
- 새 아키텍처를 처음부터 만든 게 아니라, **"결정적 회귀 모델"을 "diffusion denoiser"로 용도 전환**
- 달라진 부분
    1. **입력에 timestep 임베딩이 더해짐** — "지금이 노이즈 제거 과정의 몇 번째 단계인지"를 모델에 알려줌 (위치 인코딩과 같은 sinusoidal 방식이지만, 대상이 "시간축 위치"가 아니라 "노이즈 단계")
    2. **입력에 "관측여부" 플래그 채널이 추가됨** — 그 프레임이 컨텍스트/목표/보너스 키프레임인지, 아니면 예측해야 할 gap인지 표시
 

**구현 내용**

1. x0 예측(노이즈가 아니라): MDM/CondMDI 계열의 특징을 그대로 따라, 노이즈 ε이 아니라 원본 회전값(x0)을 직접 예측하도록 설계했어.

2. Cosine noise schedule: 학습 때 정답에 노이즈를 섞는 방식으로 표준 cosine beta schedule을 씀 (1000스텝 기준).

3. Inpainting 방식 조건화(CondMDI 핵심 메커니즘): 매 학습 스텝마다

4. 학습 손실: x0 예측값과 정답 사이의 단일 L1, 전체 시퀀스에 대해 (SILK 트랙과 동일한 철학 — 마스킹된 gap 구간만이 아니라 전체 시퀀스 기준).

5. 추론(샘플링): 순수 1000스텝 순차 diffusion은 비현실적으로 느려서(배치당 forward 1000회), DDIM 스타일로 50스텝만 균일 간격으로 골라 밟는 서브샘플링을 적용 — 매 스텝 관측 구간을 그 시점에 맞게 다시 노이즈 낀 정답으로 재치환하면서 진행.

**성능**

T=5: L2Q=0.0258 L2P=0.0014 NPSS=0.0040 (n=3840)
T=10: L2Q=0.0566 L2P=0.0031 NPSS=0.0232 (n=3840)
T=20: L2Q=0.0919 L2P=0.0051 NPSS=0.0613 (n=3840)
T=30: L2Q=0.1178 L2P=0.0067 NPSS=0.0909 (n=3840)


https://github.com/user-attachments/assets/02f18bfb-378b-4ae0-a5b3-e28c55033729



## v4. Diffusion SILK 2 (diffusion_silk_notebook_keyframe.ipynb)

**구현 내용**

- 이전 실험에 keyframe 구조 강화

**성능**

T=5: L2Q=0.0255 L2P=0.0014 NPSS=0.0040 (n=3840)
T=10: L2Q=0.0516 L2P=0.0029 NPSS=0.0197 (n=3840)
T=20: L2Q=0.0868 L2P=0.0051 NPSS=0.0538 (n=3840)
T=30: L2Q=0.1119 L2P=0.0066 NPSS=0.0825 (n=3840)

https://github.com/user-attachments/assets/aa00aeb9-dd2c-4e8d-b7a4-1e8120fbee6c


## v5. Diffusion SILK 3 (diffusion_silk_notebook_vel_loss.ipynb)

**구현 내용**

- 속도 손실 추가

**성능**

T=5: L2Q=0.0276 L2P=0.0015 NPSS=0.0050 (n=3840)
T=10: L2Q=0.0587 L2P=0.0032 NPSS=0.0290 (n=3840)
T=20: L2Q=0.1061 L2P=0.0058 NPSS=0.0897 (n=3840)
T=30: L2Q=0.1372 L2P=0.0074 NPSS=0.1308 (n=3840)


https://github.com/user-attachments/assets/7b1a2c81-830e-4caa-ac5d-22c0cadbbb3f




## v6. Flow matching SILK (flow_silk_notebook.ipynb)

**구현 내용**

- DDPM을 flow matching으로 변경
- 샘플링 시 n_step=20

**성능**

T=5: L2Q=0.0199 L2P=0.0011 NPSS=0.0037 (n=3840)
T=10: L2Q=0.0415 L2P=0.0023 NPSS=0.0179 (n=3840)
T=20: L2Q=0.0727 L2P=0.0041 NPSS=0.0489 (n=3840)
T=30: L2Q=0.1018 L2P=0.0057 NPSS=0.0808 (n=3840)

https://github.com/user-attachments/assets/db986186-7ff1-4e12-851c-397d11190e52

