import pandas as pd
import joblib

from valorant_git import get_puuid_by_riot_id

# -----------------------------
# Riot ID → PUUID 변환
# -----------------------------
def riot_ids_to_puuids(riot_ids):

    puuids = []

    for riot in riot_ids:

        name, tag = riot.split("#")

        puuid = get_puuid_by_riot_id(name, tag)

        if puuid is None:
            raise ValueError(f"플레이어를 찾을 수 없습니다 : {riot}")

        puuids.append(puuid)

    return puuids

# -----------------------------
# 모델 및 Feature 로드
# -----------------------------
model = joblib.load("../models/xgboost_valorant.pkl")
feature_columns = joblib.load("../models/feature_columns.pkl")

# 과거 플레이어 데이터
player_df = pd.read_csv("../data/player_prematch_stats.csv")


# -----------------------------
# 최근 경기(5경기 기준) 평균 Feature 추출
# -----------------------------
def get_player_feature(puuid):
    player = player_df[player_df["puuid"] == puuid]

    if len(player) == 0:
        return None

    latest = player.sort_values("started_at").iloc[-1]

    return {
        "recent_acs": latest["recent_acs"],
        "recent_kd": latest["recent_kd"],
        "recent_kast": latest["recent_kast"],
        "recent_headshot_pct": latest["recent_headshot_pct"],
        "recent_winrate": latest["recent_winrate"],
        "agent": latest["agent"]
    }


# -----------------------------
# Duelist 판별
# -----------------------------
duelist_agents = {
    "Jett",
    "Reyna",
    "Phoenix",
    "Neon",
    "Raze",
    "Yoru",
    "Iso",
    "Waylay"
}


# -----------------------------
# Team Feature 생성
# -----------------------------
def build_team_feature(blue_puuids, red_puuids):

    blue = [get_player_feature(x) for x in blue_puuids]
    red = [get_player_feature(x) for x in red_puuids]

    if None in blue or None in red:
        raise ValueError("존재하지 않는 PUUID 포함")

    row = {}

    numeric = [
        "recent_acs",
        "recent_kd",
        "recent_kast",
        "recent_headshot_pct",
        "recent_winrate"
    ]

    for col in numeric:

        blue_mean = sum(x[col] for x in blue) / 5
        red_mean = sum(x[col] for x in red) / 5

        row[f"blue_{col}"] = blue_mean
        row[f"red_{col}"] = red_mean
        row[f"diff_{col}"] = blue_mean - red_mean

    blue_duel = sum(x["agent"] in duelist_agents for x in blue)
    red_duel = sum(x["agent"] in duelist_agents for x in red)

    row["blue_duelist_count"] = blue_duel
    row["red_duelist_count"] = red_duel
    row["diff_duelist_count"] = blue_duel - red_duel

    return pd.DataFrame([row])[feature_columns]


# -----------------------------
# 예측 함수
# -----------------------------
def predict_match(blue_team, red_team):

    blue_puuids = riot_ids_to_puuids(blue_team)
    red_puuids = riot_ids_to_puuids(red_team)

    X = build_team_feature(
        blue_puuids,
        red_puuids
    )

    prob = model.predict_proba(X)[0][1]

    print("=" * 40)
    print(f"Blue 승리 확률 : {prob*100:.1f}%")

    if prob >= 0.5:
        print("예상 승리 : BLUE")
    else:
        print("예상 승리 : RED")

    print("=" * 40)

    return prob


# -----------------------------
# 예시
# -----------------------------
if __name__ == "__main__":

    blue = [
        "name 과 tag 로 입력",
        "name 과 tag 로 입력",
        "name 과 tag 로 입력",
        "name 과 tag 로 입력",
        "name 과 tag 로 입력",
    ]

    red = [
        "name 과 tag 로 입력",
        "name 과 tag 로 입력",
        "name 과 tag 로 입력",
        "name 과 tag 로 입력",
        "name 과 tag 로 입력",
    ]

    predict_match(blue, red)