import pandas as pd

df = pd.read_csv(
    "../data/player_match_stats_sample.csv"
)

df["started_at"] = pd.to_datetime(
    df["started_at"],
    format="ISO8601",
    utc=True
)

df = df.sort_values(
    ["puuid", "started_at"]
).reset_index(drop=True)

# Rolling Feature 생성
WINDOW = 5

features = [
    "acs",
    "kd",
    "headshot_pct",
    "kast",
    "win"
]

for col in features:

    rolling = (
        df.groupby("puuid")[col]
          .transform(
              lambda s: (
                  s.shift(1)
                   .rolling(
                       WINDOW,
                       min_periods=1
                   )
                   .mean()
              )
          )
    )

    if col == "win":
        df["recent_winrate"] = rolling
    else:
        df[f"recent_{col}"] = rolling

df = (
    df[df["recent_acs"].notna()]
      .reset_index(drop=True)
)

# 생성 결과 확인
player = df["puuid"].iloc[0]

check = (
    df[df["puuid"] == player]
      [["started_at",
        "acs",
        "recent_acs",
        "win",
        "recent_winrate"]]
)

df.to_csv(
    "../data/player_prematch_stats.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df.shape)
print(df["recent_acs"].notna().sum())

print(df[
    ["recent_acs",
     "recent_kd",
     "recent_winrate"]
].isna().sum())

print(df["match_id"].nunique())

print(df.head())