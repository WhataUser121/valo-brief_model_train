import pandas as pd

# 원본 (10명 모두 존재)
raw_df = pd.read_csv("../data/player_match_stats_sample.csv")

# Rolling Feature
prematch_df = pd.read_csv("../data/player_prematch_stats.csv")

# 시간 정렬
raw_df["started_at"] = pd.to_datetime(
    raw_df["started_at"],
    format="ISO8601",
    utc=True
)
prematch_df["started_at"] = pd.to_datetime(
    prematch_df["started_at"],
    format="ISO8601",
    utc=True
)

raw_df = raw_df.sort_values("started_at").reset_index(drop=True)
prematch_df = prematch_df.sort_values("started_at").reset_index(drop=True)

# -----------------------------
# 사용할 Feature
# -----------------------------
numeric_features = [
    "recent_acs",
    "recent_kd",
    "recent_kast",
    "recent_headshot_pct",
    "recent_winrate"
]

recent_cols = [
    "match_id",
    "puuid",
    "recent_acs",
    "recent_kd",
    "recent_kast",
    "recent_headshot_pct",
    "recent_winrate"
]

df = raw_df.merge(
    prematch_df[recent_cols],
    on=["match_id", "puuid"],
    how="left",
    indicator=True

)

# -----------------------------
# 역할 분류
# -----------------------------
duelists = [
    "Jett",
    "Reyna",
    "Phoenix",
    "Raze",
    "Yoru",
    "Neon",
    "Iso",
    "Waylay"
]

df["is_duelist"] = (
    df["agent"]
      .isin(duelists)
      .astype(int)
)


# -----------------------------
# 경기별 집계
# -----------------------------
rows = []

for match_id, match in df.groupby("match_id"):

    blue = match[match["team"] == "Blue"]
    red = match[match["team"] == "Red"]

    # 인원이 부족하면 제외 = 5 vs 5 경기만 사용
    if (
        len(blue) != 5 or
        len(red) != 5 or
        blue["puuid"].nunique() != 5 or
        red["puuid"].nunique() != 5
    ):
        continue

    blue_recent = blue.dropna(subset=["recent_acs"])
    red_recent = red.dropna(subset=["recent_acs"])

    # 최소 3명 이상 recent 정보가 있어야 사용
    if len(blue_recent) < 3 or len(red_recent) < 3:
        continue

    row = {
        "match_id": match_id,
        "started_at": match["started_at"].iloc[0],
        "target": int(blue["win"].iloc[0])
    }

    for col in numeric_features:
        blue_mean = blue_recent[col].mean()
        red_mean = red_recent[col].mean()

        row[f"blue_{col}"] = blue_mean
        row[f"red_{col}"] = red_mean
        row[f"diff_{col}"] = blue_mean - red_mean

    blue_duel = blue["is_duelist"].sum()
    red_duel = red["is_duelist"].sum()

    row["blue_duelist_count"] = blue_duel
    row["red_duelist_count"] = red_duel
    row["diff_duelist_count"] = blue_duel - red_duel

    rows.append(row)

team_df = pd.DataFrame(rows)

# -----------------------------
# 결측치 확인
# -----------------------------
print("\n결측치 개수")
print(team_df.isna().sum())

# -----------------------------
# 저장
# -----------------------------
team_df.to_csv(
    "../data/team_features.csv",
    index=False,
    encoding="utf-8-sig"
)

# -----------------------------
# 최종 검증 코드
# -----------------------------
print(team_df.shape)
print(team_df["target"].value_counts())
print(team_df.isna().sum().sum())