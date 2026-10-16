"""Build per-game MLB 2025, NFL 2025 and NHL 2025-26 files (+ xy CSVs).

Sources (official league APIs are blocked from this environment, so GitHub mirrors are used):
  MLB: sportsdataverse/baseballr-data  -> raw MLB StatsAPI feed/live per game
  NHL: sportsdataverse/fastRhockey-nhl-raw -> raw NHL api-web gamecenter per game (+ parsed pbp)
  NFL: nflverse play_by_play_2025
"""
import csv, gzip, io, json, re, sys, time, unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

OUT = Path("/home/user/pipeline")
RAW = "https://raw.githubusercontent.com/sportsdataverse"
S = requests.Session()


def get(url, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"failed {url}")


def slug(s):
    return re.sub(r"[^A-Za-z0-9]", "_", str(s))


def dump(path, obj):
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- MLB
MLB_XY = ["game_pk", "date", "inning", "half", "at_bat_index", "pitch_number", "batter_id", "batter",
          "bat_side", "pitcher_id", "pitcher", "pitch_hand", "balls", "strikes", "outs", "call",
          "pitch_type", "pitch_type_desc", "start_speed", "end_speed", "spin_rate", "spin_direction",
          "zone", "sz_top", "sz_bot", "plate_x", "plate_z", "gameday_x", "gameday_y", "x0", "y0", "z0",
          "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "extension",
          "is_in_play", "event", "launch_speed", "launch_angle", "total_distance", "trajectory",
          "hardness", "hit_location", "hit_coord_x", "hit_coord_y", "play_id"]


def mlb_xy(g, date):
    rows = []
    for p in g["liveData"]["plays"]["allPlays"]:
        a, m = p["about"], p["matchup"]
        for e in p.get("playEvents", []):
            if not e.get("isPitch"):
                continue
            d, pd_, c = e.get("details", {}), e.get("pitchData", {}), e.get("count", {})
            co, br, hd = pd_.get("coordinates", {}), pd_.get("breaks", {}), e.get("hitData", {})
            rows.append({
                "game_pk": g["gamePk"], "date": date, "inning": a.get("inning"), "half": a.get("halfInning"),
                "at_bat_index": a.get("atBatIndex"), "pitch_number": e.get("pitchNumber"),
                "batter_id": m["batter"]["id"], "batter": m["batter"].get("fullName"),
                "bat_side": m.get("batSide", {}).get("code"), "pitcher_id": m["pitcher"]["id"],
                "pitcher": m["pitcher"].get("fullName"), "pitch_hand": m.get("pitchHand", {}).get("code"),
                "balls": c.get("balls"), "strikes": c.get("strikes"), "outs": c.get("outs"),
                "call": d.get("description"), "pitch_type": d.get("type", {}).get("code"),
                "pitch_type_desc": d.get("type", {}).get("description"),
                "start_speed": pd_.get("startSpeed"), "end_speed": pd_.get("endSpeed"),
                "spin_rate": br.get("spinRate"), "spin_direction": br.get("spinDirection"),
                "zone": pd_.get("zone"), "sz_top": pd_.get("strikeZoneTop"), "sz_bot": pd_.get("strikeZoneBottom"),
                "plate_x": co.get("pX"), "plate_z": co.get("pZ"), "gameday_x": co.get("x"), "gameday_y": co.get("y"),
                "x0": co.get("x0"), "y0": co.get("y0"), "z0": co.get("z0"), "pfx_x": co.get("pfxX"),
                "pfx_z": co.get("pfxZ"), "vx0": co.get("vX0"), "vy0": co.get("vY0"), "vz0": co.get("vZ0"),
                "ax": co.get("aX"), "ay": co.get("aY"), "az": co.get("aZ"), "extension": pd_.get("extension"),
                "is_in_play": d.get("isInPlay"),
                "event": p["result"].get("event") if d.get("isInPlay") else None,
                "launch_speed": hd.get("launchSpeed"), "launch_angle": hd.get("launchAngle"),
                "total_distance": hd.get("totalDistance"), "trajectory": hd.get("trajectory"),
                "hardness": hd.get("hardness"), "hit_location": hd.get("location"),
                "hit_coord_x": hd.get("coordinates", {}).get("coordX"),
                "hit_coord_y": hd.get("coordinates", {}).get("coordY"), "play_id": e.get("playId"),
            })
    return rows


def mlb():
    base = OUT / "MLB 2025"
    (base / "xy").mkdir(parents=True, exist_ok=True)
    sched = pd.read_parquet(io.BytesIO(get(f"{RAW}/baseballr-data/main/mlb/schedule/2025.parquet")))
    sched = sched[sched.game_type.isin(list("RFDLW")) & sched.status_code.str.startswith("F")]
    pks = sorted(set(sched.game_pk))
    print("MLB games", len(pks), flush=True)

    def one(pk):
        blob = get(f"{RAW}/baseballr-data/main/mlb/raw/2025/{pk}.json.gz")
        if blob is None:
            return pk, None
        g = json.loads(gzip.decompress(blob))
        gd = g["gameData"]
        date = gd["datetime"].get("officialDate")
        name = f"{date}_{slug(gd['teams']['home']['name'])}_vs_{slug(gd['teams']['away']['name'])}_{pk}"
        dump(base / f"{name}.json", {"gamePk": pk, "gameData": gd, "liveData": g["liveData"]})
        write_csv(base / "xy" / f"{name}_pitches_xy.csv", mlb_xy(g, date), MLB_XY)
        return pk, name

    run(one, pks, "MLB")


# ---------------------------------------------------------------- NHL
NHL_XY = ["game_id", "event_idx", "event_id", "period", "period_type", "period_time", "game_seconds",
          "event_type", "event", "secondary_type", "event_team_abbr", "event_team_type", "description",
          "event_player_1_name", "event_player_1_id", "event_player_1_type", "event_player_2_name",
          "event_player_2_id", "event_player_2_type", "event_player_3_name", "event_player_3_id",
          "event_player_3_type", "event_goalie_name", "event_goalie_id", "strength_state", "empty_net",
          "home_score", "away_score", "x", "y", "x_fixed", "y_fixed", "shot_distance", "shot_angle", "xg"]
NHL_KEEP = ["game_info", "pbp_raw", "boxscore_raw", "landing_raw", "rosters", "shifts", "scoring",
            "penalties", "linescore", "team_box", "skater_stats", "goalie_stats"]


def nhl():
    base = OUT / "NHL 2025-2026"
    (base / "xy").mkdir(parents=True, exist_ok=True)
    sched = pd.read_parquet(io.BytesIO(get(
        f"{RAW}/fastRhockey-nhl-data/main/nhl/schedules/parquet/nhl_schedule_2026.parquet")))
    sched = sched[sched.game_state.isin(["OFF", "FINAL"])].set_index("game_id")
    print("NHL games", len(sched), flush=True)

    def one(gid):
        r = sched.loc[gid]
        blob = get(f"{RAW}/fastRhockey-nhl-raw/main/nhl/json/final/{gid}.json")
        if blob is None:
            return gid, None
        d = json.loads(blob)
        home = d["pbp_raw"]["homeTeam"]["commonName"]["default"]
        away = d["pbp_raw"]["awayTeam"]["commonName"]["default"]
        home = f"{d['pbp_raw']['homeTeam'].get('placeName', {}).get('default', '')} {home}".strip()
        away = f"{d['pbp_raw']['awayTeam'].get('placeName', {}).get('default', '')} {away}".strip()
        name = f"{r.game_date}_{slug(home)}_vs_{slug(away)}_{gid}"
        dump(base / f"{name}.json", {"game_id": int(gid), **{k: d.get(k) for k in NHL_KEEP}})
        rows = [{k: p.get(k) for k in NHL_XY} for p in d.get("all_plays") or [] if p.get("x") is not None]
        for row in rows:
            row["game_id"] = gid
        write_csv(base / "xy" / f"{name}_events_xy.csv", rows, NHL_XY)
        return gid, name

    run(one, list(sched.index), "NHL")


# ---------------------------------------------------------------- NFL
def nfl():
    base = OUT / "NFL 2025"
    base.mkdir(parents=True, exist_ok=True)
    pbp = pd.read_parquet(io.BytesIO(get(
        "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.parquet")))
    print("NFL games", pbp.game_id.nunique(), flush=True)
    for gid, g in pbp.groupby("game_id", sort=True):
        f = g.iloc[0]
        name = f"{f.game_date}_{f.home_team}_vs_{f.away_team}_{gid}"
        game = {"game_id": gid, "old_game_id": f.old_game_id, "season": int(f.season),
                "season_type": f.season_type, "week": int(f.week), "game_date": f.game_date,
                "home_team": f.home_team, "away_team": f.away_team,
                "home_score": int(f.home_score), "away_score": int(f.away_score),
                "stadium": f.stadium, "roof": f.roof, "surface": f.surface,
                "plays": [{k: v for k, v in p.items() if v is not None}
                          for p in json.loads(g.to_json(orient="records"))]}
        dump(base / f"{name}.json", game)
    print("NFL done", flush=True)


def run(fn, ids, tag):
    done = miss = 0
    with ThreadPoolExecutor(16) as ex:
        for i, (k, name) in enumerate(ex.map(fn, ids)):
            done += name is not None
            if name is None:
                miss += 1
                print(f"{tag} missing {k}", flush=True)
            if i % 200 == 0:
                print(f"{tag} {i}/{len(ids)}", flush=True)
    print(f"{tag} done {done} missing {miss}", flush=True)


if __name__ == "__main__":
    for which in sys.argv[1:]:
        {"mlb": mlb, "nhl": nhl, "nfl": nfl}[which]()
