"""
공적마스크 5부제 약국 대기열 시뮬레이션 - Streamlit 웹앱
=============================================================
실행:
    streamlit run mask_sim_app.py

기능:
  • 사이드바 슬라이더 → 변경 즉시 자동 재시뮬레이션
  • A~F 탭별 시나리오 설명 / 핵심 코드 / 재고 그래프
  • 전체 비교 탭: 대기시간·성공률·비용·재고 4종 자동 업데이트
  • 일주일(월~금 5부제 + 토~일 이월 수요) 단위 시뮬레이션
"""

import streamlit as st
import simpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List
import matplotlib.font_manager as fm
import os

# 폰트 캐시 강제 재빌드 (Streamlit Cloud 대응)
fm._load_fontmanager(try_read_cache=False)

# 시스템에 설치된 폰트 중 한글 폰트 탐색
def get_korean_font():
    font_candidates = ['NanumGothic', 'NanumBarunGothic', 'NanumMyeongjo', 
                       'Malgun Gothic', 'AppleGothic', 'UnDotum']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in font_candidates:
        if font in available:
            return font
    return None

korean_font = get_korean_font()
if korean_font:
    plt.rcParams['font.family'] = korean_font
else:
    # 절대경로로 직접 로드 (fallback)
    nanum_paths = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
    ]
    for path in nanum_paths:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams['font.family'] = fm.FontProperties(fname=path).get_name()
            break

plt.rcParams['axes.unicode_minus'] = False


st.set_page_config(page_title="공적마스크 시뮬레이션", page_icon="💊", layout="wide")


# ═══════════════════════════════════════════════════════════════
# 1. 데이터클래스
# ═══════════════════════════════════════════════════════════════
@dataclass
class Params:
    scenario: str = "A"
    inventory: int = 100
    masks_per_person: int = 5
    n_servers: int = 1
    mu1_min: float = 0.5
    mu1_max: float = 1.0
    mu2_min: float = 5.0
    mu2_max: float = 15.0
    lambda1_peak: float = 60.0
    lambda1_bg: float = 5.0
    lambda2: float = 8.0
    use_5buje: bool = False
    eligible_fraction: float = 1.0
    app_peak_reduction: float = 0.0
    delivery_time: float = 30.0
    day_duration: float = 480.0       # 하루 영업 시간 (분)
    n_weekdays: int = 5               # 월~금
    n_weekend_days: int = 2           # 토~일
    # 주중 미충족 수요 중 주말에 재방문하는 비율
    weekend_demand_fraction: float = 0.5
    cost_per_unmet: float = 2500.0
    cost_auxiliary_day: float = 30750.0
    cost_per_mask: float = 1500.0
    n_replications: int = 30
    seed: int = 42


@dataclass
class WeekResult:
    waiting_times: List[float]  = field(default_factory=list)
    purchase_success: int       = 0
    purchase_fail: int          = 0
    server_busy_time: float     = 0.0
    max_queue_length: int       = 0
    stockout_days: int          = 0
    # 요일별 재고 타임라인 (월~일 순서로 쌓임, 그래프용)
    inventory_timeline: List    = field(default_factory=list)
    n_servers: int              = 1    # ← 이거 추가

    @property
    def avg_waiting_time(self):
        return float(np.mean(self.waiting_times)) if self.waiting_times else 0.0

    @property
    def purchase_success_rate(self):
        total = self.purchase_success + self.purchase_fail
        return self.purchase_success / total if total > 0 else 0.0

    @property
    def server_utilization(self):
        return self.server_busy_time / (480.0 * 7 * self.n_servers * 2)


# ═══════════════════════════════════════════════════════════════
# 2. 시뮬레이션 엔진 (일주일 단위)
# ═══════════════════════════════════════════════════════════════
class PharmacySim:
    """
    일주일 시뮬레이션.
    - 월~금: eligible_fraction=0.2 (5부제)
    - 토~일: 주중 미충족 수요 이월 (weekend_demand_fraction)
    - 매일 아침 재고 보충
    """
    def __init__(self, params: Params, seed: int):
        self.p   = params
        self.rng = np.random.default_rng(seed)
        self.result = WeekResult(n_servers=params.n_servers)

    def _run_one_day(self, day_idx: int, extra_lambda: float = 0.0):
        """
        하루치 시뮬레이션.
        day_idx: 0=월 ~ 4=금, 5=토, 6=일
        extra_lambda: 주말 이월 수요 도착률 추가분 (명/시간)
        반환: (day_result dict, 품절시각 or None, inv_timeline)
        """
        env = simpy.Environment()
        servers = [simpy.Resource(env, capacity=self.p.n_servers) for _ in range(2)]
        inventories = [self.p.inventory] * 2
        stockouts = [False] * 2
        stockout_time = [None]
        inv_tl = [(0.0, float(self.p.inventory))]
        day_res = {'fail': 0, 'success': 0, 'waits': [], 'busy': 0.0, 'max_q': 0}

        is_weekend = day_idx >= 5
        if is_weekend:
            eff_fraction = 1.0
        elif self.p.use_5buje:
            eff_fraction = self.p.eligible_fraction
        else:
            eff_fraction = 1.0

        def mask_customer():
            has_app = self.p.app_peak_reduction > 0.0
            target = self.rng.choice([0, 1])
            if stockouts[target]:
                if has_app:
                    alt = 1 - target
                    if not stockouts[alt]:
                        yield env.timeout(5.0)
                        target = alt
                    else:
                        day_res['fail'] += 1; return
                else:
                    day_res['fail'] += 1; return
            arrive = env.now
            q_len = len(servers[target].queue)
            if q_len > day_res['max_q']:
                day_res['max_q'] = q_len
            with servers[target].request() as req:
                yield req
                if stockouts[target] or inventories[target] < self.p.masks_per_person:
                    day_res['fail'] += 1
                    day_res['waits'].append(env.now - arrive); return
                day_res['waits'].append(env.now - arrive)
                svc = self.rng.uniform(self.p.mu1_min, self.p.mu1_max)
                day_res['busy'] += svc
                yield env.timeout(svc)
                inventories[target] -= self.p.masks_per_person
                day_res['success'] += 1
                inv_tl.append((env.now, float(inventories[0])))
                if inventories[target] <= 0 and not stockouts[target]:
                    stockouts[target] = True
                    if stockout_time[0] is None:
                        stockout_time[0] = env.now

        def general_customer():
            target = self.rng.choice([0, 1])
            with servers[target].request() as req:
                yield req
                svc = self.rng.uniform(self.p.mu2_min, self.p.mu2_max)
                day_res['busy'] += svc
                yield env.timeout(svc)

        def arrival_mask():
            while True:
                t = env.now
                if all(stockouts): return
                peak = self.p.lambda1_peak * (1 - self.p.app_peak_reduction)
                base_lam = (self.p.lambda1_bg if t < self.p.delivery_time
                            else peak * eff_fraction) / 60.0
                extra_lam = (extra_lambda / 60.0) if is_weekend else 0.0
                lam = base_lam + extra_lam
                iat = self.rng.exponential(1.0 / lam) if lam > 0 else 1e9
                yield env.timeout(iat)
                if env.now >= self.p.day_duration: return
                env.process(mask_customer())

        def arrival_general():
            lam = (self.p.lambda2 / 60.0) * 2
            while True:
                iat = self.rng.exponential(1.0 / lam)
                yield env.timeout(iat)
                if env.now >= self.p.day_duration: return
                env.process(general_customer())

        env.process(arrival_mask())
        env.process(arrival_general())
        env.run(until=self.p.day_duration)
        inv_tl.append((self.p.day_duration, max(inventories[0], 0)))
        return day_res, stockout_time[0], inv_tl

    def run(self) -> WeekResult:
        total_weekday_fail = 0

        # 월~금
        for day in range(self.p.n_weekdays):
            day_res, st, inv_tl = self._run_one_day(day_idx=day)
            self.result.waiting_times.extend(day_res['waits'])
            self.result.purchase_success += day_res['success']
            self.result.purchase_fail    += day_res['fail']
            self.result.server_busy_time += day_res['busy']
            self.result.max_queue_length  = max(
                self.result.max_queue_length, day_res['max_q'])
            # 재고 타임라인: 요일 오프셋(day * day_duration) 더해서 이어붙임
            offset = day * self.p.day_duration
            for t, v in inv_tl:
                self.result.inventory_timeline.append((offset + t, v))
            if st is not None:
                self.result.stockout_days += 1
            total_weekday_fail += day_res['fail']

        # 주말 이월 수요 계산
        weekend_extra_people = total_weekday_fail * self.p.weekend_demand_fraction
        weekend_lambda = weekend_extra_people / (self.p.n_weekend_days * 8.0)

        # 토~일
        for i, day in enumerate(range(5, 5 + self.p.n_weekend_days)):
            day_res, st, inv_tl = self._run_one_day(
                day_idx=day, extra_lambda=weekend_lambda)
            self.result.waiting_times.extend(day_res['waits'])
            self.result.purchase_success += day_res['success']
            self.result.purchase_fail    += day_res['fail']
            self.result.server_busy_time += day_res['busy']
            self.result.max_queue_length  = max(
                self.result.max_queue_length, day_res['max_q'])
            offset = (5 + i) * self.p.day_duration
            for t, v in inv_tl:
                self.result.inventory_timeline.append((offset + t, v))
            if st is not None:
                self.result.stockout_days += 1

        return self.result


# ═══════════════════════════════════════════════════════════════
# 3. 시나리오 팩토리 & 실행
# ═══════════════════════════════════════════════════════════════
def get_params(sc, lam):
    p = Params(scenario=sc, lambda1_peak=lam)
    if sc == "A":
        p.inventory, p.masks_per_person = 100, 5
        p.mu1_min, p.mu1_max = 0.5, 1.0
    elif sc == "B":
        p.inventory, p.masks_per_person = 100, 2
        p.mu1_min, p.mu1_max = 0.5, 1.0
    elif sc == "C":
        p.inventory, p.masks_per_person = 100, 2
        p.use_5buje, p.eligible_fraction = True, 0.2
        p.mu1_min, p.mu1_max = 1.0, 2.0
    elif sc == "D":
        p.inventory, p.masks_per_person = 250, 2
        p.use_5buje, p.eligible_fraction = True, 0.2
        p.mu1_min, p.mu1_max = 1.0, 2.0
    elif sc == "E":
        p.inventory, p.masks_per_person = 250, 2
        p.use_5buje, p.eligible_fraction = True, 0.2
        p.n_servers = 2
        p.mu1_min, p.mu1_max = 1.0, 2.0
    elif sc == "F":
        p.inventory, p.masks_per_person = 250, 2
        p.use_5buje, p.eligible_fraction = True, 0.2
        p.app_peak_reduction = 0.3
        p.mu1_min, p.mu1_max = 1.0, 2.0
    return p


def run_sim(params):
    rng = np.random.default_rng(params.seed)
    seeds = rng.integers(0, 99999, size=params.n_replications)
    waits, srs, stockout_days_list, mqs, utils, inv_tls = [], [], [], [], [], []

    for seed in seeds:
        sim = PharmacySim(params, int(seed))
        r = sim.run()
        waits.append(r.avg_waiting_time)
        srs.append(r.purchase_success_rate)
        stockout_days_list.append(r.stockout_days)
        mqs.append(r.max_queue_length)
        utils.append(r.server_utilization)
        inv_tls.append(r.inventory_timeline)

    avg_sr = float(np.mean(srs))
    est_arr = (params.lambda1_peak * params.eligible_fraction / 60
               * params.day_duration * params.n_weekdays)
    cost_unmet = (1 - avg_sr) * est_arr * params.cost_per_unmet
    cost_staff = (params.n_servers - 1) * params.cost_auxiliary_day * 7
    cost_inv   = params.inventory * params.cost_per_mask

    return {
        "scenario"     : params.scenario,
        "avg_wait"     : float(np.mean(waits)),
        "success_rate" : avg_sr,
        "stockout_days": float(np.mean(stockout_days_list)),
        "max_queue"    : float(np.mean(mqs)),
        "utilization"  : float(np.mean(utils)),
        "total_cost"   : cost_unmet + cost_staff + cost_inv,
        "cost_unmet"   : cost_unmet,
        "cost_staff"   : cost_staff,
        "cost_inv"     : cost_inv,
        "_inv_tls"     : inv_tls,
    }


@st.cache_data(show_spinner=False)
def cached_run_all(lam, n_rep):
    results = {}
    for sc in ["A", "B", "C", "D", "E", "F"]:
        p = get_params(sc, lam)
        p.n_replications = n_rep
        results[sc] = run_sim(p)
    return results


# ═══════════════════════════════════════════════════════════════
# 4. 시각화
# ═══════════════════════════════════════════════════════════════
COLORS = {
    "A": "#DC2626", "B": "#D97706", "C": "#2563EB",
    "D": "#059669", "E": "#7C3AED", "F": "#0D9488",
}

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

SC_META = {
    "A": {"title": "A: 무정책 (Baseline)",
          "desc": "**정책 없음** — 구매 제한도 5부제도 없음.\n\n"
                  "- 초기 재고: **100장** / 1인 구매: **5장** (사실상 무제한)\n"
                  "- DUR 없음: 서비스 시간 Uniform(0.5, 1.0) min\n"
                  "- 주말: 주중 미충족 수요 50% 이월",
          "code": "p.inventory, p.masks_per_person = 100, 5\n"
                  "p.use_5buje = False\n"
                  "p.mu1_min, p.mu1_max = 0.5, 1.0  # DUR 없음"},
    "B": {"title": "B: 1인 2매 제한",
          "desc": "**구매량만 제한** — 5부제 없이 1인 2매 제한만.\n\n"
                  "- 초기 재고: **100장** / 1인 구매: **2장**\n"
                  "- DUR 없음 → 서비스 속도 유지\n"
                  "- 재고 소진 속도 감소, 대기열 분산은 없음",
          "code": "p.inventory, p.masks_per_person = 100, 2\n"
                  "p.use_5buje = False\n"
                  "p.mu1_min, p.mu1_max = 0.5, 1.0"},
    "C": {"title": "C: 5부제 시행",
          "desc": "**5부제 도입** — 생년 끝자리 기준 요일별 구매 제한(1/5).\n\n"
                  "- 초기 재고: **100장** / eligible_fraction: **0.2**\n"
                  "- DUR 도입 → Uniform(1.0, 2.0) min\n"
                  "- **주말**: 주중 미충족 수요 50% 이월 → 토·일 수요 증가",
          "code": "p.inventory = 100\n"
                  "p.use_5buje, p.eligible_fraction = True, 0.2\n"
                  "p.mu1_min, p.mu1_max = 1.0, 2.0  # DUR 도입\n"
                  "p.weekend_demand_fraction = 0.5   # 이월 수요"},
    "D": {"title": "D: 5부제 + 공급 확대",
          "desc": "**공급량 증가** — 5부제 유지하면서 재고 250장으로 확대.\n\n"
                  "- 초기 재고: **250장** / 서버: **1명**\n"
                  "- 재고 부족은 해결, 서버 병목은 그대로\n"
                  "- 주말 이월 수요에도 재고 여유로 대응 가능",
          "code": "p.inventory = 250  # 재고 확대!\n"
                  "p.use_5buje, p.eligible_fraction = True, 0.2\n"
                  "p.n_servers = 1    # 서버 1명 유지 → 병목 미해결"},
    "E": {"title": "E: 5부제 + 보조인력",
          "desc": "**보조인력 투입** — 서버 2명으로 처리 속도 2배.\n\n"
                  "- 초기 재고: **250장** / 서버: **2명**\n"
                  "- Uniform(0.5, 1.0) min — 보조인력이 신분증 확인\n"
                  "- 대기시간 급감 → 주말 이월 수요도 빠르게 처리\n"
                  "- 추가 비용: **30,750원/일 × 7일**",
          "code": "p.inventory = 250\n"
                  "p.n_servers = 2              # 보조인력 투입!\n"
                  "p.mu1_min, p.mu1_max = 0.5, 1.0  # 처리 속도 회복"},
    "F": {"title": "F: 5부제 + 재고앱",
          "desc": "**정보 비대칭 해소** — 재고 현황 앱으로 고객 분산.\n\n"
                  "- 초기 재고: **250장** (약국 2개 네트워크)\n"
                  "- 피크 수요 **30% 감소** (app_peak_reduction=0.3)\n"
                  "- 주말: 앱 효과로 이월 수요도 분산 처리\n"
                  "- 추가 비용: **0원**",
          "code": "p.inventory = 250\n"
                  "p.app_peak_reduction = 0.3  # 앱으로 30% 수요 분산\n"
                  "# 추가 비용 0원!"},
}


def fig_inventory(inv_tls, sc, color, initial_inv):
    """일주일 재고 소진 그래프 (X축: 요일)"""
    # 7일 × 480분 = 3360분
    TIME_BINS = np.linspace(0, 3360, 337)
    curves = []
    for tl in inv_tls:
        ts  = [x[0] for x in tl]
        ivs = [x[1] for x in tl]
        if len(ts) < 2:
            curves.append(np.full(len(TIME_BINS), initial_inv, dtype=float))
            continue
        curves.append(np.interp(TIME_BINS, ts, ivs))

    avg = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    # 개별 런
    for c in curves:
        ax.plot(TIME_BINS / 480, c, color=color, alpha=0.06, linewidth=0.5)
    ax.fill_between(TIME_BINS / 480, np.maximum(avg - std, 0), avg + std,
                    color=color, alpha=0.18)
    ax.plot(TIME_BINS / 480, avg, color=color, linewidth=2.4, label="Mean")

    ax.axhline(y=initial_inv, color=color, linestyle=":", alpha=0.4, linewidth=1.1)
    ax.text(0.05, initial_inv + 4, f"Initial/day: {initial_inv}",
            fontsize=8, color=color, alpha=0.75)

    # 요일 경계선 + 매일 입고 표시
    for d in range(7):
        ax.axvline(x=d, color="#CBD5E1", linewidth=0.8, alpha=0.6)
        # 매일 입고 시점 (30분 = 0.0625일)
        ax.axvline(x=d + 30/480, color="#94A3B8", linestyle=":",
                   linewidth=0.8, alpha=0.5)

    # 주말 배경 음영
    ax.axvspan(5, 7, color="#FEF9C3", alpha=0.35, zorder=0)
    ax.text(5.05, initial_inv * 0.9, "Weekend\n(이월 수요)",
            fontsize=8, color="#92400E", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FEF9C3",
                      edgecolor="#D97706", alpha=0.85))

    ax.fill_between(TIME_BINS / 480, 0, 3, color="#EF4444", alpha=0.07)
    ax.set_xlabel("요일", fontsize=9)
    ax.set_ylabel("잔여 재고 (장)", fontsize=9)
    ax.set_title(f"Scenario {sc} — 일주일 재고 흐름 (mean ± 1σ)",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(0, 7)
    ax.set_xticks(range(8))
    ax.set_xticklabels(DAY_LABELS + [""], fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.18, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_bar(results, metric, title, ylabel):
    scs  = list(results.keys())
    vals = [results[sc][metric] for sc in scs]
    clrs = [COLORS[sc] for sc in scs]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")
    bars = ax.bar(scs, vals, color=clrs, edgecolor="white",
                  linewidth=1.2, width=0.6)
    for bar, val in zip(bars, vals):
        fmt = f"{val:,.0f}" if (val > 1000 or "cost" in metric) else f"{val:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01, fmt,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.22 if max(vals) > 0 else 1)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def fig_inventory_all(results):
    """전체 비교 재고 그래프 (일주일, A·C·D·E)"""
    TIME_BINS = np.linspace(0, 3360, 337)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    for sc in ["A", "C", "D", "E"]:
        inv_tls = results[sc]["_inv_tls"]
        color   = COLORS[sc]
        initial = get_params(sc, 60).inventory
        curves  = []
        for tl in inv_tls:
            ts  = [x[0] for x in tl]
            ivs = [x[1] for x in tl]
            if len(ts) < 2:
                curves.append(np.full(len(TIME_BINS), initial, dtype=float))
                continue
            curves.append(np.interp(TIME_BINS, ts, ivs))
        avg = np.mean(curves, axis=0)
        std = np.std(curves, axis=0)
        ax.fill_between(TIME_BINS / 480, np.maximum(avg - std, 0), avg + std,
                        color=color, alpha=0.12)
        ax.plot(TIME_BINS / 480, avg, color=color,
                linewidth=2.3, label=f"Scenario {sc}")

    for d in range(7):
        ax.axvline(x=d, color="#CBD5E1", linewidth=0.8, alpha=0.5)
    ax.axvspan(5, 7, color="#FEF9C3", alpha=0.3, zorder=0)
    ax.text(5.05, 10, "Weekend", fontsize=8, color="#92400E")
    ax.fill_between(TIME_BINS / 480, 0, 3, color="#EF4444", alpha=0.06)

    ax.set_xlabel("요일", fontsize=9)
    ax.set_ylabel("잔여 재고 (장)", fontsize=9)
    ax.set_title("일주일 재고 흐름 비교 A·C·D·E (mean ± 1σ)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 7)
    ax.set_xticks(range(8))
    ax.set_xticklabels(DAY_LABELS + [""], fontsize=8)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.18, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# 5. UI
# ═══════════════════════════════════════════════════════════════
st.title("💊 공적마스크 5부제 약국 대기열 시뮬레이션")
st.caption("M(t)/G/c/FCFS + Finite Inventory + Multiple Customer Classes | 일주일 단위 시뮬레이션")

with st.sidebar:
    st.header("⚙️ 공통 파라미터")
    st.caption("슬라이더 조절 → 전체 자동 업데이트")
    lambda1_peak = st.slider(
        "λ_peak (피크 도착률, 명/시간)", 20, 150, 60, 10,
        help="입고 직후 마스크 구매 고객의 시간당 도착 수")
    n_rep = st.slider(
        "반복 횟수 (replications)", 10, 80, 30, 10,
        help="높을수록 정확하나 느림. 발표용 30 권장")
    st.divider()
    st.info(
        f"**현재 설정**\n\n"
        f"λ_peak = **{lambda1_peak}** 명/시간\n\n"
        f"반복 = **{n_rep}** 회\n\n"
        f"*슬라이더 바꾸면 즉시 재계산*"
    )

with st.spinner(f"시뮬레이션 실행 중 (λ={lambda1_peak}, rep={n_rep})..."):
    all_results = cached_run_all(lambda1_peak, n_rep)

tabs = st.tabs([
    "A: 무정책", "B: 1인2매", "C: 5부제",
    "D: 5부제+공급", "E: 5부제+인력", "F: 5부제+앱", "📊 전체 비교"
])

for sc_key, tab in zip(["A", "B", "C", "D", "E", "F"], tabs[:6]):
    meta  = SC_META[sc_key]
    color = COLORS[sc_key]
    res   = all_results[sc_key]
    p     = get_params(sc_key, lambda1_peak)

    with tab:
        col_desc, col_code = st.columns([1, 1])
        with col_desc:
            st.subheader(meta["title"])
            st.markdown(meta["desc"])
        with col_code:
            st.markdown("**핵심 코드**")
            st.code(meta["code"], language="python")

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⏱ 평균 대기시간", f"{res['avg_wait']:.1f} 분")
        c2.metric("✅ 구매 성공률",   f"{res['success_rate']*100:.1f} %")
        c3.metric("📦 품절 발생 일수", f"{res['stockout_days']:.1f} 일 / 주")
        c4.metric("💰 총 비용 (주간)", f"{res['total_cost']:,.0f} 원")

        st.divider()

        fig_inv = fig_inventory(
            res["_inv_tls"], sc_key, color, initial_inv=p.inventory)
        st.pyplot(fig_inv)
        plt.close(fig_inv)

        with st.expander("💴 비용 상세"):
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("미충족 손실", f"{res['cost_unmet']:,.0f} 원")
            bc2.metric("인력 비용",   f"{res['cost_staff']:,.0f} 원")
            bc3.metric("재고 비용",   f"{res['cost_inv']:,.0f} 원")

with tabs[6]:
    st.subheader("📊 시나리오 A~F 전체 비교")
    st.caption(
        f"λ_peak = {lambda1_peak} 명/시간 | 반복 = {n_rep}회 | "
        f"일주일 단위 | *슬라이더 바꾸면 자동 업데이트*"
    )

    rows = []
    for sc in ["A", "B", "C", "D", "E", "F"]:
        r = all_results[sc]
        rows.append({
            "시나리오":       sc,
            "평균 대기(분)":  f"{r['avg_wait']:.1f}",
            "성공률":         f"{r['success_rate']*100:.1f}%",
            "품절 일수/주":   f"{r['stockout_days']:.1f}일",
            "최대 대기열(명)":f"{r['max_queue']:.1f}",
            "서버 이용률":    f"{r['utilization']:.2f}",
            "총 비용(원)":    f"{r['total_cost']:,.0f}",
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("시나리오"),
        use_container_width=True)
    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        f1 = fig_bar(all_results, "avg_wait",
                     "⏱ 평균 대기시간 (분)  ↓ 낮을수록 좋음", "분")
        st.pyplot(f1); plt.close(f1)
        f2 = fig_bar(all_results, "success_rate",
                     "✅ 구매 성공률  ↑ 높을수록 좋음", "비율 (0~1)")
        st.pyplot(f2); plt.close(f2)
    with col_r:
        f3 = fig_bar(all_results, "total_cost",
                     "💰 총 비용 (원, 주간)  ↓ 낮을수록 좋음", "원 (KRW)")
        st.pyplot(f3); plt.close(f3)
        f4 = fig_inventory_all(all_results)
        st.pyplot(f4); plt.close(f4)

    st.divider()
    st.subheader("💡 핵심 발견")
    best_wait = min(["A","B","C","D","E","F"],
                    key=lambda s: all_results[s]["avg_wait"])
    best_cost = min(["A","B","C","D","E","F"],
                    key=lambda s: all_results[s]["total_cost"])
    best_sr   = max(["A","B","C","D","E","F"],
                    key=lambda s: all_results[s]["success_rate"])
    st.markdown(f"""
| 지표 | 최우수 시나리오 | 값 |
|------|----------------|----|
| 최단 대기시간 | **{best_wait}** | {all_results[best_wait]['avg_wait']:.1f} 분 |
| 최저 총비용   | **{best_cost}** | {all_results[best_cost]['total_cost']:,.0f} 원 |
| 최고 성공률   | **{best_sr}**   | {all_results[best_sr]['success_rate']*100:.1f} % |

> **핵심 결론**: 서버 처리 속도(μ)가 핵심 병목.
> 재고 확대(D)만으로는 대기열 해소 불가 →
> **보조인력(E)**가 대기시간·성공률 모두 최적.
> 예산 제약 시 **재고앱(F)**이 무비용 대안.
""")