import argparse
import json
import os
import sys
import numpy as np

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


DEFAULT_FILE   = "FO FINAL DATASET.xlsx"
SHEET_NAME     = "FO_Master_Dataset"
INDEX_PATH     = "./fo_faiss.index"        # FAISS vector index
METADATA_PATH  = "./fo_metadata.json"      # record metadata + documents
EMBED_MODEL    = "all-MiniLM-L6-v2"        


def clean(val) -> str:
    if val is None or (isinstance(val, float) and str(val) == 'nan'):
        return ""
    return str(val).strip()


def build_document(row) -> str:
    name        = clean(row.get("FO_Name"))
    fo_type     = clean(row.get("FO_Type"))
    family      = clean(row.get("Founding_Family"))
    wealth      = clean(row.get("Wealth_Source"))
    year        = clean(row.get("Year_Founded"))
    city        = clean(row.get("HQ_City"))
    country     = clean(row.get("HQ_Country"))
    region      = clean(row.get("Region"))
    aum         = clean(row.get("AUM_USD_Millions_Est"))
    website     = clean(row.get("Website"))
    dm_name     = clean(row.get("Primary_Decision_Maker"))
    dm_title    = clean(row.get("Title_Primary_DM"))
    dm2_name    = clean(row.get("Secondary_Decision_Maker"))
    dm2_title   = clean(row.get("Title_Secondary_DM"))
    linkedin    = clean(row.get("LinkedIn_Primary_DM"))
    email       = clean(row.get("General_Email"))
    phone       = clean(row.get("Phone"))
    strategy    = clean(row.get("Investment_Strategy"))
    sectors     = clean(row.get("Sector_Focus"))
    geo         = clean(row.get("Geographic_Focus"))
    chk_min     = clean(row.get("Check_Size_Min_USD_M"))
    chk_max     = clean(row.get("Check_Size_Max_USD_M"))
    co_invest   = clean(row.get("Co_Invest_Appetite"))
    asset_class = clean(row.get("Asset_Class_Preference"))
    portfolio   = clean(row.get("Portfolio_Companies_Examples"))
    funds       = clean(row.get("Notable_Fund_Relationships"))
    esg         = clean(row.get("ESG_Impact_Focus"))
    succession  = clean(row.get("Succession_Stage"))
    news        = clean(row.get("Recent_Signal_News_2023_2025"))
    confidence  = clean(row.get("Data_Confidence"))
    sources     = clean(row.get("Primary_Source"))

    doc_parts = []

    fo_desc = f"{name} is a {fo_type}"
    if family:   fo_desc += f" managed by the {family}"
    if wealth:   fo_desc += f", with wealth originating from {wealth}"
    if year:     fo_desc += f". Founded in {year}"
    if city and country: fo_desc += f", headquartered in {city}, {country} ({region})"
    doc_parts.append(fo_desc + ".")

    if aum:
        try:
            aum_val = float(aum)
            aum_str = f"${aum_val/1000:.1f}B" if aum_val >= 1000 else f"${aum_val:.0f}M"
            doc_parts.append(f"Estimated AUM: {aum_str} USD.")
        except:
            doc_parts.append(f"Estimated AUM: {aum} million USD.")

    if dm_name:
        dm_line = f"Primary decision maker: {dm_name}"
        if dm_title:  dm_line += f" ({dm_title})"
        if linkedin:  dm_line += f". LinkedIn: {linkedin}"
        if email:     dm_line += f". Email: {email}"
        if phone:     dm_line += f". Phone: {phone}"
        doc_parts.append(dm_line + ".")
    if dm2_name:
        dm2_line = f"Secondary decision maker: {dm2_name}"
        if dm2_title: dm2_line += f" ({dm2_title})"
        doc_parts.append(dm2_line + ".")

    if strategy:    doc_parts.append(f"Investment strategy: {strategy}.")
    if sectors:     doc_parts.append(f"Sector focus: {sectors}.")
    if geo:         doc_parts.append(f"Geographic focus: {geo}.")
    if asset_class: doc_parts.append(f"Asset class preference: {asset_class}.")

    if chk_min and chk_max:
        doc_parts.append(f"Typical check size: ${chk_min}M to ${chk_max}M USD.")

    if co_invest:   doc_parts.append(f"Co-investment appetite: {co_invest}.")
    if portfolio:   doc_parts.append(f"Notable portfolio companies: {portfolio}.")
    if funds:       doc_parts.append(f"Fund relationships: {funds}.")
    if esg:         doc_parts.append(f"ESG and impact focus: {esg}.")
    if succession:  doc_parts.append(f"Succession stage: {succession}.")
    if news:        doc_parts.append(f"Recent activity (2023-2025): {news}")
    if website:     doc_parts.append(f"Website: {website}.")

    doc_parts.append(f"Data confidence: {confidence}. Sources: {sources}.")

    return " ".join(doc_parts)


def build_metadata(row) -> dict:
    def sf(val, default=0.0):
        try:
            v = float(val)
            return v if str(v) != 'nan' else default
        except:
            return default

    return {
        "record_id":      clean(row.get("Record_ID"))              or "unknown",
        "fo_name":        clean(row.get("FO_Name"))                or "unknown",
        "fo_type":        clean(row.get("FO_Type"))                or "unknown",
        "region":         clean(row.get("Region"))                 or "unknown",
        "country":        clean(row.get("HQ_Country"))             or "unknown",
        "aum_millions":   sf(row.get("AUM_USD_Millions_Est")),
        "check_min":      sf(row.get("Check_Size_Min_USD_M")),
        "check_max":      sf(row.get("Check_Size_Max_USD_M")),
        "co_invest":      clean(row.get("Co_Invest_Appetite"))     or "unknown",
        "esg":            clean(row.get("ESG_Impact_Focus"))       or "unknown",
        "succession":     clean(row.get("Succession_Stage"))       or "unknown",
        "confidence":     clean(row.get("Data_Confidence"))        or "unknown",
        "sectors":        clean(row.get("Sector_Focus"))           or "unknown",
        "strategy":       clean(row.get("Investment_Strategy"))    or "unknown",
        "decision_maker": clean(row.get("Primary_Decision_Maker")) or "unknown",
        "website":        clean(row.get("Website"))                or "N/A",
        "email":          clean(row.get("General_Email"))          or "N/A",
        "linkedin":       clean(row.get("LinkedIn_Primary_DM"))    or "N/A",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  default=DEFAULT_FILE)
    parser.add_argument("--sheet", default=SHEET_NAME)
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  FO RAG PIPELINE — STEP 1: INGEST")
    print("=" * 60)

    print(f"\n  Loading dataset: {args.file}")
    df = pd.read_excel(args.file, sheet_name=args.sheet)
    print(f"  Records found : {len(df)}")

    print(f"\n  Building document chunks...")
    documents = []
    metadatas = []
    ids       = []

    for _, row in df.iterrows():
        doc  = build_document(row)
        meta = build_metadata(row)
        rid  = clean(row.get("Record_ID")) or f"FO-{_:03d}"
        documents.append(doc)
        metadatas.append(meta)
        ids.append(rid)

    print(f"  Documents built : {len(documents)}")
    print(f"  Avg doc length  : {sum(len(d) for d in documents)//len(documents)} chars")

    print(f"\n  Loading embedding model: {EMBED_MODEL}")
    print(f"  (First run downloads ~90MB model)")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"\n  Generating embeddings for {len(documents)} records...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)

    print(f"\n  Building FAISS index...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner product = cosine similarity after normalisation
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"  FAISS index saved: {INDEX_PATH}  ({index.ntotal} vectors)")

    
    store = {"ids": ids, "documents": documents, "metadatas": metadatas}
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)
    print(f"  Metadata saved : {METADATA_PATH}")

    print(f"\n  Ingestion complete!")
    print(f"  Total vectors  : {index.ntotal}")
    print()
    print("  Run step2_query.py to start querying.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
