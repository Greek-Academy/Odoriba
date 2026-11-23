"""運搬者ありの余裕を、どの項が決めているかを返す limiting_term のテスト。

家全体のスキャンで箱200cmの余裕が -83cm と出た。壁に83cmめり込んでいるの
ではなく、手の高さが届く範囲から83cm外れている値だった。値だけでは区別
できないので、項の名前を一緒に返す。
"""

import numpy as np

import phase2_lstair as L
import scan_plan as sp


class _OpenSpace:
    """壁が遠くにしか無い、平らな床だけの空間(床の高さは0)。"""

    def clearance_points(self, pts):
        return np.full(len(pts), 100.0)

    def floor_fn(self, x, y, z_hint=None):
        return 0.0


def _pose(z):
    return (np.array([0.0, 0.0, z]), L._pose(0.0, 0.0).as_quat())


def test_reach_is_reported_when_hands_are_too_high(monkeypatch):
    monkeypatch.setattr(L, "FURN_L", 100.0)
    monkeypatch.setattr(L, "FURN_W", 45.0)
    monkeypatch.setattr(L, "FURN_H", 60.0)
    name, terms = sp.limiting_term(_OpenSpace(), _pose(300.0), num_carriers=2)
    assert name == "reach"
    assert abs(terms["reach"] - (L.REACH_MAX - 300.0)) < 1e-6   # 手が届く上限より高い
    assert terms["furniture"] > 0                                # 壁には当たっていない


def test_terms_cover_every_enabled_oracle_term(monkeypatch):
    monkeypatch.setattr(L, "FURN_L", 100.0)
    monkeypatch.setattr(L, "FURN_W", 45.0)
    monkeypatch.setattr(L, "FURN_H", 60.0)
    name, terms = sp.limiting_term(_OpenSpace(), _pose(100.0), num_carriers=2)
    assert set(terms) == {"furniture", "reach", "body", "self"}
    assert terms[name] == min(terms.values())
    assert terms["reach"] > 0                                    # 腰の高さなら届く
