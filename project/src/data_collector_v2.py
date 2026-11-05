import json
import pandas as pd
from collections import deque
from pathlib import Path

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed
)
from valorant_git import (
    extract_player_rows_from_match,
    get_match_detail_v4,
    get_puuid_by_riot_id,
    get_matches_v4,
    PLATFORM
)

DATA_DIR = Path("../data")

HISTORY_FILE = DATA_DIR / "player_histories.json"
CHECKPOINT = DATA_DIR / "checkpoint_players.json"
MATCH_CACHE = DATA_DIR / "match_cache.json"

# ===== Config =====
TARGET_PLAYERS = 50
TOTAL_MATCHES = 50
REGION = "kr"
PLATFORM = "pc"

def load_checkpoint():
    if not CHECKPOINT.exists():
        return set()

    with open(CHECKPOINT, "r") as f:

        data = json.load(f)

    return set(data["completed"])


def save_checkpoint(completed):
    CHECKPOINT.parent.mkdir(exist_ok=True)

    with open(CHECKPOINT, "w") as f:

        json.dump(
            {
                "completed": list(completed)
            },
            f,
            indent=2
        )

def save_histories(histories):

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(
            histories,
            f,
            ensure_ascii=False,
            indent=2
        )

def load_histories():
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, FileNotFoundError):
        return []

def collect_unique_puuids(seed_puuids, region="kr", platform="pc", total_matches=20, target_players=150):
    queue = deque(seed_puuids)
    visited = set(seed_puuids)

    match_cache = {}

    while queue and len(visited) < target_players:

        puuid = queue.popleft()

        matches = get_matches_v4(
            region,
            platform,
            puuid,
            total_matches
        )

        for match in matches:

            match_id = (
                match.get("metadata", {}).get("match_id")
            )

            if not match_id:
                continue

            if match_id in match_cache:
                detail = match_cache[match_id]
            else:
                detail = get_match_detail_v4(
                    region,
                    match_id
                )

                if detail is None:
                    continue

                match_cache[match_id] = detail

            players = detail.get("players", [])

            for p in players:

                if len(visited) >= target_players:
                    break

                new_puuid = p.get("puuid")

                if (
                    new_puuid
                    and new_puuid not in visited
                    and len(visited) < target_players
                ):
                    visited.add(new_puuid)
                    queue.append(new_puuid)

            if len(visited) >= target_players:
                break

        # print(
        #     f"Players : {len(visited)} / {target_players} | "
        #     f"Queue : {len(queue)}"
        # )

    return list(visited)

def collect_unique_match_ids(histories):

    print(histories[0].keys())

    match_ids = set()

    for history in histories:
        for match in history["matches"]:
            match_ids.add(
                match["metadata"]["match_id"]
            )

    return sorted(match_ids)

# 데이터 품질 향상을 위해 모든 플레이어의 puuid 를 갖고 게임 정보 조회 -> 수십시간 걸림. 현실적으로 불가능 하여 기능 롤백함.
# def extract_all_puuids(df):
#     """
#     player_match_stats에서 등장하는 모든 플레이어 PUUID 추출
#     """
#     puuids = (
#         df["puuid"]
#         .dropna()
#         .drop_duplicates()
#         .tolist()
#     )

#     print("전체 PUUID :", len(puuids))

#     return puuids

def collect_player_histories(unique_puuids):

    completed = load_checkpoint()

    histories = load_histories()

    # 중복 저장 방지
    history_map = {
        h["puuid"]: h
        for h in histories
    }

    total = len(unique_puuids)

    for idx, puuid in enumerate(unique_puuids, start=1):

        if puuid in completed:
            continue

        matches = get_matches_v4(
            REGION,
            PLATFORM,
            puuid,
            TOTAL_MATCHES
        )

        history_map[puuid] = {
            "puuid": puuid,
            "matches": matches
        }

        completed.add(puuid)

        save_histories(
            list(history_map.values())
        )

        save_checkpoint(completed)

        print(
            f"[{idx}/{total}] 완료"
        )

    return list(history_map.values())

def load_match_cache():

    if not MATCH_CACHE.exists():
        return {}

    with open(MATCH_CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)

    # 🔹 모든 키를 소문자로 통일
    return {k.lower(): v for k, v in cache.items()}

def download_match_cache(match_ids, region="kr"):

    cache = load_match_cache()

    remaining = [
        m for m in match_ids
        if m not in cache
    ]

    MATCH_CACHE.write_text(
        json.dumps(cache)
    )

    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = {
            executor.submit(
                get_match_detail_v4,
                region,
                match_id
            ): match_id

            for match_id in remaining
        }

        total = len(futures)

        for idx, future in enumerate(as_completed(futures), 1):

            match_id = futures[future]

            detail = future.result()

            if detail:
                cache[match_id] = detail

            print(
                f"[Match] {idx}/{total}",
                end="\r"
            )

    with open(MATCH_CACHE, "w", encoding="utf-8") as f:
        json.dump(
            cache,
            f,
            ensure_ascii=False,
            indent=2
        )
    
    return cache

def build_dataset(histories, match_cache):
    rows = []

    for history in histories:

        puuid = history["puuid"]

        for match in history["matches"]:

            match_id = (
                match["metadata"]["match_id"]
                .lower()
            )

            detail = match_cache.get(match_id)

            if detail is None:
                print("MISS :", match_id)
                continue

            if not detail:
                continue

            player_rows = extract_player_rows_from_match(
                detail,
                puuid
            )

            rows.extend(player_rows)

    df = pd.DataFrame(rows)

    # 같은 경기의 같은 플레이어는 1개만 유지
    df = (
        df.drop_duplicates(
            subset=["match_id", "puuid"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    return df

def run(seed_players):

    # 1. Seed 경기
    seed_puuids = []

    for player in seed_players:

        puuid = get_puuid_by_riot_id(
            player["name"],
            player["tag"]
        )

        if puuid:
            seed_puuids.append(puuid)

    # 2. 모든 플레이어
    unique_puuids = collect_unique_puuids(
        seed_puuids=seed_puuids,
        total_matches=TOTAL_MATCHES,
        target_players=TARGET_PLAYERS
    )

    # print("Seed PUUID :", len(seed_puuids))
    # print("최종 플레이어 :", len(unique_puuids))

    # 3. 병렬 수집
    histories = collect_player_histories(unique_puuids)

    print("Unique PUUID :", len(unique_puuids))
    print("Histories :", len(histories))

    # 4. Match Cache
    unique_matches = collect_unique_match_ids(histories)

    cache = download_match_cache(unique_matches)

    # 5. Dataset
    df = build_dataset(histories, cache)

    # # 5. ⭐ 모든 플레이어 PUUID 추출
    # all_puuids = extract_all_puuids(df)

    # # 6. ⭐ 모든 플레이어 Histories 생성
    # histories = collect_player_histories(all_puuids)

    # # 7. CSV 다시 생성 (최종)
    # match_cache = download_match_cache(histories)
    # df = build_dataset(histories, match_cache)

    print(df.shape)
    print(df["match_id"].nunique())
    print(df["puuid"].nunique())
    print(df.duplicated(
        subset=["match_id","puuid"]
    ).sum())

    # 6. csv 저장
    df.to_csv(
        "../data/player_match_stats.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"완료 : {len(df)} rows")

    return df

if __name__ == "__main__":

    seed_players = [
        {
            "name": "-",
            "tag": "-"
        }
    ]

    df = run(seed_players)
