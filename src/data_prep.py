import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
# Correct import path (Astroquery moved this module)
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive as EA


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_hashes(paths, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {p: sha256_file(p) for p in paths if os.path.isfile(p)}
    if out_file.exists():
        try:
            existing = json.loads(out_file.read_text())
        except Exception:
            existing = {}
        existing.update(payload)
        out_file.write_text(json.dumps(existing, indent=2))
    else:
        out_file.write_text(json.dumps(payload, indent=2))


def _ensure_schema(df: pd.DataFrame, wanted: list) -> pd.DataFrame:
    for c in wanted:
        if c not in df.columns:
            df[c] = pd.NA
    return df[wanted].copy()


def fetch_exoplanet_archive(out_dir: str) -> list:
    """
    Confirmed planets from 'ps' (composite parameters) and candidates from 'toi'
    where disposition-like column is not 'CONFIRMED'. We discover the actual
    TOI disposition column at runtime (e.g., 'tfopwg_disp', 'disposition', ...).
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "pscomppars_sample.csv")

    wanted = [
        "pl_name", "pl_orbper", "pl_rade", "st_teff", "st_rad",
        "sy_snum", "sy_pnum", "disc_year", "discoverymethod"
    ]
    num_cols = ["pl_orbper", "pl_rade", "st_teff", "st_rad", "sy_snum", "sy_pnum"]

    try:
        # -------- Confirmed planets (PS composite) --------
        ps = EA.query_criteria(
            table="ps",
            select="pl_name,pl_orbper,pl_rade,st_teff,st_rad,sy_snum,sy_pnum,disc_year,discoverymethod"
        ).to_pandas()
        ps = _ensure_schema(ps, wanted)
        ps = ps.drop_duplicates(subset=["pl_name"]).reset_index(drop=True)
        ps["target_confirmed"] = 1

        # -------- Candidates (TOI) --------
        toi_raw = EA.query_criteria(table="toi", select="*").to_pandas()

        # Find a usable disposition column and filter out confirmed
        disp_col = None
        for c in ["disposition", "tfopwg_disp", "tfopwg_disposition", "TFOPWG_DISP", "TFOPWG_DISPOSITION"]:
            if c in toi_raw.columns:
                disp_col = c
                break
        if disp_col:
            mask = toi_raw[disp_col].astype(str).str.upper() != "CONFIRMED"
            toi_raw = toi_raw[mask].copy()

        # Map common TOI fields into our schema
        alias = {
            "toi": "pl_name",
            "pl_orbper": "pl_orbper", "orbper": "pl_orbper",
            "pl_rade": "pl_rade", "rad": "pl_rade",
            "st_teff": "st_teff",
            "st_rad": "st_rad",
            "sy_snum": "sy_snum",
            "sy_pnum": "sy_pnum",
            "disc_year": "disc_year",
            "discoverymethod": "discoverymethod",
        }
        toi = toi_raw.copy()
        for src, dst in alias.items():
            if src in toi.columns:
                toi[dst] = toi[src]
        toi = _ensure_schema(toi, wanted)

        # Backfill discovery year from any available date-ish column
        if toi["disc_year"].isna().all():
            for col in ["disc_pubdate", "disc_discovered", "date", "created"]:
                if col in toi_raw.columns:
                    toi["disc_year"] = pd.to_datetime(toi_raw[col], errors="coerce").dt.year
                    break

        # TOIs are transit candidates by construction if missing
        toi["discoverymethod"] = toi["discoverymethod"].fillna("Transit")
        toi["target_confirmed"] = 0

        # -------- Combine & minimal cleaning --------
        df = pd.concat([ps, toi], ignore_index=True)
        df = df.dropna(subset=["pl_orbper"]).reset_index(drop=True)

        # Impute numeric from confirmed medians
        med = ps[num_cols].apply(pd.to_numeric, errors="coerce").median()
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med[c])

        if df["disc_year"].isna().any():
            df["disc_year"] = df["disc_year"].fillna(df["disc_year"].median())

        # Balance classes for a stable starter set
        n1 = int((df["target_confirmed"] == 1).sum())
        n0 = int((df["target_confirmed"] == 0).sum())
        if min(n1, n0) == 0:
            raise ValueError(f"One class missing after processing. confirmed={n1}, candidates={n0}")
        target_n = max(400, min(n1, n0))
        pos = df[df["target_confirmed"] == 1].sample(n=min(target_n, n1), random_state=42)
        neg = df[df["target_confirmed"] == 0].sample(n=min(target_n, n0), random_state=42)
        df_bal = (pd.concat([pos, neg], ignore_index=True)
                  .drop_duplicates(subset=["pl_name"])
                  .sample(frac=1.0, random_state=42)
                  .reset_index(drop=True))

        df_bal.to_csv(csv_path, index=False)
        return [csv_path]

    except Exception as e:
        # Always produce a tiny working sample
        print(f"Warning: falling back to tiny sample due to: {e}")
        df = pd.DataFrame({
            "pl_name": [
                "Kepler-10 b", "Kepler-22 b", "Kepler-444 e",
                "TESS-1234.01", "TESS-5678.01", "TESS-9012.01"
            ],
            "pl_orbper": [0.837, 289.8623, 6.189, 4.21, 12.8, 1.93],
            "pl_rade":   [1.47, 2.38, 0.76, 2.10, 1.15, 3.00],
            "st_teff":   [5627, 5518, 5046, 5900, 4800, 6100],
            "st_rad":    [1.07, 0.98, 0.75, 0.95, 0.70, 1.10],
            "sy_snum":   [1, 1, 1, 1, 1, 1],
            "sy_pnum":   [2, 1, 5, 1, 1, 1],
            "disc_year": [2011, 2011, 2015, 2022, 2022, 2023],
            "discoverymethod": ["Transit"] * 6,
            "target_confirmed": [1, 1, 1, 0, 0, 0],
        })
        df.to_csv(csv_path, index=False)
        return [csv_path]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--out", type=str, default="data/raw")
    parser.add_argument("--hash", type=str, default="artifacts/data_hash.json")
    args = parser.parse_args()

    saved = []
    if args.fetch:
        saved = fetch_exoplanet_archive(args.out)

    if saved:
        write_hashes(saved, args.hash)
        print(f"Saved and hashed: {saved}")
    else:
        print("No files fetched.")


if __name__ == "__main__":
    main()