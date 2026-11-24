"""候補の一括判定(scan_candidates)のテスト。

掃引そのものは scan_plan.judge に任せているので、ここでは「余裕をどう
○/△/×に分けるか」「寸法の差し替えが後に残らないか」「1件の失敗で残りが
止まらないか」を、judge を差し替えて確かめる。
"""

import pytest

import phase2_lstair as L
import scan_candidates as sc
import scan_plan as sp


class _Centerline:
    """弧長 100〜200cm だけが曲がる所の中心線。"""

    s_total = 500.0

    def corner_at(self, s):
        return (100.0, 200.0) if 100.0 <= s <= 200.0 else None

    def xy_at(self, s):
        return [float(s), 0.0]


class _Space:
    centerline = _Centerline()


def _result(capacity, s=50.0, no_pose=(), term="furniture", part="furniture",
            terms=None, others=()):
    """judge が返す dict のうち、一括判定が読む部分だけ。

    others は、余裕が最小の位置のほかに掃引した位置 [(s, 余裕), ...]。
    """
    b = {"s": s, "capacity_cm": capacity, "xy": [0.0, 0.0], "no_pose_s": list(no_pose),
         "contact": {"xyz": [0.0, 0.0, 0.0], "clearance_cm": capacity, "part": part},
         "profile": [[s, capacity]] + [list(o) for o in others]}
    return {"bottleneck_furniture_only": dict(b, capacity_cm=capacity + 5.0),
            "bottleneck": dict(b, limit={"term": term,
                                         "terms": terms or {term: capacity}})}


# ---- classify ----

@pytest.mark.parametrize("capacity, at_corner, expected", [
    (3.0, False, "ok"),       # 直線部分は±3cm
    (2.9, False, "close"),
    (-2.9, False, "close"),
    (-3.0, False, "ng"),
    (8.9, True, "close"),     # 曲がる所は±9cm
    (9.0, True, "ok"),
    (-8.9, True, "close"),
    (-9.0, True, "ng"),
])
def test_classify_uses_the_scan_error_of_the_place(capacity, at_corner, expected):
    assert sc.classify(capacity, at_corner) == expected


def test_classify_no_pose_is_ng_even_with_room():
    assert sc.classify(20.0, False, no_pose=True) == "ng"


def test_classify_out_of_reach_is_ng_not_close():
    # 手が届かない量は壁との距離ではないので、誤差の幅で「際どい」にしない
    assert sc.classify(-1.0, False, term="reach") == "ng"
    assert sc.classify(10.0, False, term="reach") == "ok"


# ---- furniture_dims ----

def test_furniture_dims_sorts_the_horizontal_sides():
    with sc.furniture_dims(40.0, 90.0, 180.0):
        assert (L.FURN_L, L.FURN_W, L.FURN_H) == (90.0, 40.0, 180.0)


def test_furniture_dims_restores_after_an_error():
    orig = (L.FURN_L, L.FURN_W, L.FURN_H)
    with pytest.raises(RuntimeError):
        with sc.furniture_dims(10.0, 20.0, 30.0):
            raise RuntimeError("途中で失敗")
    assert (L.FURN_L, L.FURN_W, L.FURN_H) == orig


# ---- orientations ----

def test_orientations_default_keeps_the_height_up():
    assert sc.orientations(180.0, 85.0, 80.0) == [(180.0, 85.0, 80.0)]


def test_orientations_any_side_tries_each_side_up_once():
    ups = [o[2] for o in sc.orientations(180.0, 85.0, 80.0, any_side=True)]
    assert sorted(ups) == [80.0, 85.0, 180.0]
    # 立方体はどの向きも同じなので1通り
    assert len(sc.orientations(50.0, 50.0, 50.0, any_side=True)) == 1


# ---- judge_candidates ----

def test_judge_candidates_keeps_going_after_one_failure(monkeypatch):
    def fake_judge(space, **kwargs):
        if L.FURN_L >= 300.0:
            raise ValueError("経路が短すぎて掃引できない")
        return _result(12.0)

    monkeypatch.setattr(sp, "judge", fake_judge)
    orig = (L.FURN_L, L.FURN_W, L.FURN_H)
    rows = sc.judge_candidates(_Space(), [("長すぎ", 300.0, 50.0, 50.0),
                                          ("小", 60.0, 40.0, 40.0)], log=lambda *_: None)
    assert [r["verdict"] for r in rows] == ["unknown", "ok"]
    assert "短すぎ" in rows[0]["error"]
    assert (L.FURN_L, L.FURN_W, L.FURN_H) == orig


def test_judge_candidates_passes_the_sizes_to_the_oracle(monkeypatch):
    seen = []

    def fake_judge(space, **kwargs):
        seen.append((L.FURN_L, L.FURN_W, L.FURN_H, kwargs["num_carriers"], kwargs["rrt"]))
        return _result(12.0)

    monkeypatch.setattr(sp, "judge", fake_judge)
    sc.judge_candidates(_Space(), [("棚", 40.0, 90.0, 180.0)], num_carriers=1,
                        log=lambda *_: None)
    assert seen == [(90.0, 40.0, 180.0, 1, False)]


def test_judge_candidates_any_side_takes_the_best_orientation(monkeypatch):
    def fake_judge(space, **kwargs):
        # 180cmの辺を上下にしたときだけ余裕がある
        return _result(15.0 if L.FURN_H == 180.0 else -20.0)

    monkeypatch.setattr(sp, "judge", fake_judge)
    cand = [("棚", 90.0, 180.0, 40.0)]
    as_is = sc.judge_candidates(_Space(), cand, log=lambda *_: None)[0]
    any_side = sc.judge_candidates(_Space(), cand, any_side=True, log=lambda *_: None)[0]
    assert as_is["verdict"] == "ng"
    assert any_side["verdict"] == "ok"
    assert any_side["carried_as"] == [90.0, 40.0, 180.0]
    assert "180cmの辺を上下" in sc.format_row(any_side)


def test_judge_candidates_uses_the_corner_error_at_a_corner(monkeypatch):
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result(5.0, s=150.0))
    row = sc.judge_candidates(_Space(), [("箱", 100.0, 45.0, 60.0)], log=lambda *_: None)[0]
    assert row["at_corner"] and row["verdict"] == "close"   # 直線部分なら○になる余裕


# ---- 表示 ----

def test_shortage_text_does_not_show_reach_as_centimeters(monkeypatch):
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result(-83.0, term="reach"))
    row = sc.judge_candidates(_Space(), [("箱", 200.0, 50.0, 65.0)], log=lambda *_: None)[0]
    assert row["verdict"] == "ng"
    assert sc.shortage_text(row) == "手が届かない"


def test_shortage_text_says_how_much_is_missing(monkeypatch):
    monkeypatch.setattr(sp, "judge",
                        lambda space, **kw: _result(-4.0, s=150.0, term="body"))
    row = sc.judge_candidates(_Space(), [("箱", 100.0, 45.0, 60.0)], log=lambda *_: None)[0]
    text = sc.shortage_text(row)
    assert "あと4cm足りない" in text and "曲がる所" in text and "運ぶ人と壁の間" in text


def test_shortage_text_names_the_limiting_term_not_the_deepest_point(monkeypatch):
    # 一番めり込んだ点が家具でも、余裕を決めたのが運搬者と家具の重なりならそう言う
    monkeypatch.setattr(sp, "judge",
                        lambda space, **kw: _result(-7.0, term="self", part="furniture"))
    row = sc.judge_candidates(_Space(), [("棚", 90.0, 40.0, 180.0)], log=lambda *_: None)[0]
    assert "運ぶ人と家具の間" in sc.shortage_text(row)


# ---- 判定を決める位置 ----

def test_verdict_is_ng_when_a_straight_part_is_short_beyond_its_error(monkeypatch):
    # 最小は曲がる所の-5cm(±9cmの内側)だが、直線の-4cmは±3cmの外で入らない
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result(
        -5.0, s=150.0, others=[(300.0, -4.0)]))
    row = sc.judge_candidates(_Space(), [("箱", 100.0, 45.0, 60.0)], log=lambda *_: None)[0]
    assert row["verdict"] == "ng"
    assert (row["s"], row["capacity_cm"], row["at_corner"]) == (300.0, -4.0, False)
    assert row["min_capacity_cm"] == -5.0
    assert "直線部分" in sc.shortage_text(row)


def test_verdict_is_close_when_a_corner_is_inside_its_error(monkeypatch):
    # 最小は直線の+4cm(○)だが、曲がる所の+6cmは±9cmの内側で際どい
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result(
        4.0, s=300.0, others=[(150.0, 6.0)]))
    row = sc.judge_candidates(_Space(), [("箱", 100.0, 45.0, 60.0)], log=lambda *_: None)[0]
    assert row["verdict"] == "close"
    assert (row["s"], row["at_corner"]) == (150.0, True)


def test_positive_reach_is_not_shown_as_room_to_the_wall(monkeypatch):
    # 手が届く範囲の余り(+2cm)は壁との距離ではない。壁との距離(+25cm)で判定する
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result(
        2.0, term="reach", terms={"reach": 2.0, "furniture": 25.0, "body": 30.0}))
    row = sc.judge_candidates(_Space(), [("箱", 100.0, 45.0, 60.0)], log=lambda *_: None)[0]
    assert row["verdict"] == "ok" and row["term"] == "furniture"
    assert "余裕25cm" in sc.shortage_text(row)


# ---- CSV ----

def test_read_csv_reports_rows_it_could_not_read(tmp_path):
    path = tmp_path / "c.csv"
    lines = ["名前,幅,奥行,高さ", "ソファ,180,85,80", "棚,90cm,40,180"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    items, skipped = sc.read_csv(path)
    assert items == [("ソファ", 180.0, 85.0, 80.0)]
    assert skipped == ["棚,90cm,40,180"]


# ---- 何kgまで運べるか ----

def _result_by_tilt(by_tilt, max_tilt=55.0, **kwargs):
    """傾きの上限ごとの余裕 [(上限, 余裕), ...] を持つ judge の結果。

    今の設定での余裕は、max_tilt 以下で一番大きい上限のものにする。
    """
    now = [c for t, c in by_tilt if t <= max_tilt][-1]
    result = _result(now, **kwargs)
    result["bottleneck"]["max_tilt_deg"] = max_tilt
    result["bottleneck_by_tilt"] = [
        dict(_result(c, **kwargs)["bottleneck"], max_tilt_deg=t) for t, c in by_tilt]
    return result


_TILTS_PASS_FROM_30 = [(0.0, -20.0), (15.0, -20.0), (30.0, 12.0), (45.0, 12.0),
                       (55.0, 12.0), (70.0, 30.0), (85.0, 30.0), (90.0, 30.0)]


def test_weight_limit_comes_from_the_smallest_tilt_that_passes(monkeypatch):
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result_by_tilt(_TILTS_PASS_FROM_30))
    row = sc.judge_candidates(_Space(), [("箱", 100.0, 45.0, 60.0)], log=lambda *_: None)[0]
    assert row["verdict"] == "ok" and row["needed_tilt_deg"] == 30.0
    # 下側の人の割合 = 1/2 + (高さ/2·長さ)·tan30° = 0.673。業者40kg・自分たち25kgをこれで割る
    assert row["weight_limit_kg"]["mover"] == pytest.approx(40.0 / 0.6732, abs=0.05)
    assert row["weight_limit_kg"]["diy"] == pytest.approx(25.0 / 0.6732, abs=0.05)
    assert "[業者なら59kgまで / 自分たちなら37kgまで]" in sc.format_row(row)


def test_weight_limit_uses_the_size_of_the_candidate(monkeypatch):
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result_by_tilt(_TILTS_PASS_FROM_30))
    rows = sc.judge_candidates(_Space(), [("短い", 100.0, 45.0, 60.0),
                                          ("長い", 200.0, 45.0, 60.0)], log=lambda *_: None)
    # 同じ傾きでも、長い家具ほど下側の人に寄る重さが小さい
    assert rows[1]["weight_limit_kg"]["mover"] > rows[0]["weight_limit_kg"]["mover"]


def test_weight_limit_is_flat_carry_when_no_tilt_is_needed(monkeypatch):
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result_by_tilt(
        [(0.0, 12.0), (55.0, 12.0), (90.0, 12.0)]))
    row = sc.judge_candidates(_Space(), [("箱", 100.0, 45.0, 60.0)], log=lambda *_: None)[0]
    assert row["needed_tilt_deg"] == 0.0
    assert row["weight_limit_kg"] == {"mover": 80.0, "diy": 50.0}   # 2人で半分ずつ


def test_weight_limit_is_not_given_for_a_candidate_that_does_not_fit(monkeypatch):
    monkeypatch.setattr(sp, "judge", lambda space, **kw: _result_by_tilt(
        [(0.0, -20.0), (55.0, -20.0), (90.0, -20.0)]))
    row = sc.judge_candidates(_Space(), [("箱", 100.0, 45.0, 60.0)], log=lambda *_: None)[0]
    assert row["verdict"] == "ng" and "weight_limit_kg" not in row
    assert "kgまで" not in sc.format_row(row)
