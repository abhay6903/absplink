import os
import io
import json
import uuid
import time
import shutil
import threading
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from flask import Flask, request, jsonify, send_file, render_template, Response
import trino
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lower, trim, regexp_replace, lit, concat_ws, coalesce, monotonically_increasing_id, expr, length, row_number, size, when
from pyspark.sql.types import StringType, DoubleType
from pyspark.sql import Window
from pyspark.ml.feature import Tokenizer, HashingTF, MinHashLSH

from graphframes import GraphFrame
from concurrent.futures import ThreadPoolExecutor

# Splink (Spark)
from splink import Linker
from splink import SparkAPI
from splink.backends.spark import similarity_jar_location

# Auto-blocking module from local file
import auto_blocking2 as ab
import pandas as pd
from rapidfuzz.distance import JaroWinkler
from pyspark.storagelevel import StorageLevel
from threading import Lock
from trino.dbapi import connect
import boto3
from botocore.client import Config
from pathlib import Path
import pyarrow.parquet as pq
import trino
import config

cache_lock = Lock()
cached_linkers: Dict[str, Linker] = {}
cached_roles: Dict[str, Dict[str, Any]] = {}
cached_df_enhanced: Dict[str, SparkDataFrame] = {}
cached_report_df: Dict[str, SparkDataFrame] = {}
cached_total_rows: Dict[str, int] = {}

# ----------------------------
# App and global stats
# ----------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Local path to your Scala UDF similarity jar from notes.ipynb
CUSTOM_JAR_PATH = os.path.join(BASE_DIR, "scala-udf-similarity-0.1.1-shaded.jar")
TRINO_JAR_PATH = os.path.join(BASE_DIR, "trino-jdbc-438.jar")
# In-memory job tracking
jobs: Dict[str, Dict[str, Any]] = {}

# Connection/session state
spark: Optional[SparkSession] = None
connection: Dict[str, Any] = {
    "connected": False,
    "trino": None,  # dict of host, port, catalog, schema, user
}

# Simple session id (frontend expects one)
SESSION_ID = uuid.uuid4().hex[:16]
current_session_id = SESSION_ID

# ----------------------------
# Spark helpers
# ----------------------------
import os
from pyspark.sql import SparkSession

def get_or_create_spark() -> SparkSession:
    global spark
    if spark is not None:
        return spark

    spark_temp_dir = os.path.join(BASE_DIR, "spark-temp")
    os.makedirs(spark_temp_dir, exist_ok=True)
    spark_local_dir = os.path.join(spark_temp_dir, "local_dirs")
    os.makedirs(spark_local_dir, exist_ok=True)

    # --- Spark session configured with Hive Metastore ---
    spark = (
        SparkSession.builder.appName("SplinkSparkApp")
        #.enableHiveSupport()  # Enable Hive support
        .config("spark.driver.memory", "12g")
        .config("spark.executor.memory", "8g")
        .config("spark.python.worker.memory", "4g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.sql.codegen.wholeStage", "false")
        .config("spark.local.dir", spark_local_dir)
        .config("spark.shuffle.file.buffer", "64k")
        .config("spark.shuffle.io.maxRetries", "10")
        .config("spark.shuffle.io.retryWait", "60s")
        .config("spark.network.timeout", "800s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.ui.port", "4040")
        .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.2-s_2.12")
        .config("spark.jars", f"{similarity_jar_location()},{CUSTOM_JAR_PATH},{TRINO_JAR_PATH}")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
    # --- Spark checkpoint dir ---
    checkpoint_dir = os.path.join(BASE_DIR, "tmp_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    spark.sparkContext.setCheckpointDir(checkpoint_dir)

    # --- Register Java UDFs ---
    try:
        from pyspark.sql.types import StringType, DoubleType, ArrayType
        spark.udf.registerJavaFunction("accent_remove", "uk.gov.moj.dash.linkage.AccentRemover", StringType())
        spark.udf.registerJavaFunction("double_metaphone", "uk.gov.moj.dash.linkage.DoubleMetaphone", StringType())
        spark.udf.registerJavaFunction("double_metaphone_alt", "uk.gov.moj.dash.linkage.DoubleMetaphoneAlt", StringType())
        spark.udf.registerJavaFunction("cosine_distance_custom", "uk.gov.moj.dash.linkage.CosineDistance", DoubleType())
        spark.udf.registerJavaFunction("jaccard_similarity", "uk.gov.moj.dash.linkage.JaccardSimilarity", DoubleType())
        spark.udf.registerJavaFunction("jaro_similarity", "uk.gov.moj.dash.linkage.JaroSimilarity", DoubleType())
        spark.udf.registerJavaFunction("jaro_winkler_similarity", "uk.gov.moj.dash.linkage.JaroWinklerSimilarity", DoubleType())
        spark.udf.registerJavaFunction("lev_damerau_distance", "uk.gov.moj.dash.linkage.LevDamerauDistance", DoubleType())
        spark.udf.registerJavaFunction("qgram_tokeniser", "uk.gov.moj.dash.linkage.QgramTokeniser", StringType())
        spark.udf.registerJavaFunction("q2gram_tokeniser", "uk.gov.moj.dash.linkage.Q2gramTokeniser", StringType())
        spark.udf.registerJavaFunction("q3gram_tokeniser", "uk.gov.moj.dash.linkage.Q3gramTokeniser", StringType())
        spark.udf.registerJavaFunction("q4gram_tokeniser", "uk.gov.moj.dash.linkage.Q4gramTokeniser", StringType())
        spark.udf.registerJavaFunction("q5gram_tokeniser", "uk.gov.moj.dash.linkage.Q5gramTokeniser", StringType())
        spark.udf.registerJavaFunction("q6gram_tokeniser", "uk.gov.moj.dash.linkage.Q6gramTokeniser", StringType())
        spark.udf.registerJavaFunction("dual_array_explode", "uk.gov.moj.dash.linkage.DualArrayExplode", ArrayType(StringType()))
        spark.udf.registerJavaFunction("latlong_explode", "uk.gov.moj.dash.linkage.latlongexplode", ArrayType(StringType()))
        spark.udf.registerJavaFunction("sql_escape", "uk.gov.moj.dash.linkage.sqlEscape", StringType())
    except Exception:
        pass

    return spark


def trino_jdbc_url(cfg: Dict[str, Any]) -> str:
    host = cfg.get("host", "localhost")
    port = str(cfg.get("port", 8080))
    catalog = cfg.get("catalog", "system")
    return f"jdbc:trino://{host}:{port}/{catalog}"

def _get_trino_connection(cfg: Dict[str, Any]):
    """Create a Trino DB-API connection for metadata queries."""
    return trino.dbapi.connect(
        host=cfg.get("host"),
        port=int(cfg.get("port", 8080)),
        user=str(cfg.get("user", config.TRINO_DEFAULT_USER)),
        catalog=cfg.get("catalog"),
        schema=cfg.get("schema") or None,
    )

def read_trino_table_as_spark_df(cfg: Dict[str, Any], table: str) -> SparkDataFrame:
    sp = get_or_create_spark()
    url = trino_jdbc_url(cfg)
    props = {
        "driver": "io.trino.jdbc.TrinoDriver",
        "user": cfg.get("user", config.TRINO_DEFAULT_USER),
    }
    parts = table.split(".")
    if len(parts) == 1:
        if not cfg.get("schema"):
            raise RuntimeError(
                "Schema not set. Provide table as 'schema.table' or 'catalog.schema.table', "
                "or call /run with {'schema': '...', 'table': '...'}"
            )
        fq_table = f"{cfg['catalog']}.{cfg['schema']}.{parts[0]}"
    elif len(parts) == 2:
        fq_table = f"{cfg['catalog']}.{parts[0]}.{parts[1]}"
    else:
        fq_table = table

    return (
        sp.read.format("jdbc")
        .option("url", url)
        .option("dbtable", fq_table)
        .options(**props)
        .load()
    )

def query_trino_as_df(cfg: Dict[str, Any], sql_text: str) -> SparkDataFrame:
    sp = get_or_create_spark()
    url = trino_jdbc_url(cfg)
    props = {
        "driver": "io.trino.jdbc.TrinoDriver",
        "user": cfg.get("user", config.TRINO_DEFAULT_USER),
    }
    wrapped = f"( {sql_text} ) t"
    return (
        sp.read.format("jdbc")
        .option("url", url)
        .option("dbtable", wrapped)
        .options(**props)
        .load()
    )

def write_single_csv(
    df: SparkDataFrame, 
    out_path: str, 
    sort_cols: Optional[List[str]] = None
) -> None:
    """
    Safely write Spark DataFrame to a single CSV file, with optional sorting.
    Uses repartition(1) and combines sorting to optimize performance.
    """
    temp_dir = out_path + "__tmp__"
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    seen, renamed_cols = {}, []
    for c in df.columns:
        base = c.split(".")[-1]
        if base in seen:
            seen[base] += 1
            new_name = f"{base}_dup{seen[base]}"
            renamed_cols.append(col(c).alias(new_name))
        else:
            seen[base] = 0
            renamed_cols.append(col(c).alias(base))
    df = df.select(*renamed_cols)

    writer = df.repartition(1)
    if sort_cols:
        writer = writer.sortWithinPartitions(*sort_cols)
    
    writer.write.mode("overwrite").option("header", True).csv(temp_dir)

    part_file = None
    for root, dirs, files in os.walk(temp_dir):
        for name in files:
            if name.startswith("part-") and name.endswith(".csv"):
                part_file = os.path.join(root, name)
                break
        if part_file:
            break

    if not part_file:
        raise RuntimeError("CSV part file not generated.")

    shutil.move(part_file, out_path)
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"✅ CSV written: {out_path}")

def _purge_outputs_dir():
    """Remove files for current session to keep outputs clean."""
    if not os.path.isdir(OUTPUTS_DIR):
        return
    for name in os.listdir(OUTPUTS_DIR):
        path = os.path.join(OUTPUTS_DIR, name)
        try:
            if os.path.isfile(path) and current_session_id in name:
                os.remove(path)
            elif os.path.isdir(path) and current_session_id in name:
                shutil.rmtree(path, ignore_errors=True)
        except Exception:
            pass

# ----------------------------
# Helper functions similar to app1.py
# ----------------------------
def ensure_first_last_from_name(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record)
    name = (rec.get("name") or rec.get("full_name") or "").strip()
    if name and (not rec.get("first_name") or not rec.get("last_name")):
        parts = [p for p in name.split(" ") if p]
        if len(parts) >= 2:
            rec.setdefault("first_name", parts[0])
            rec.setdefault("last_name", parts[-1])
    return rec

def ensure_name_aliases(record: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(record)
    if rec.get("first_name") and rec.get("last_name") and not rec.get("full_name"):
        rec["full_name"] = f"{rec['first_name']} {rec['last_name']}"
    if rec.get("name") and not rec.get("full_name"):
        rec["full_name"] = rec["name"]
    return rec

def _pick_cols(columns: List[str], roles: Dict[str, str]) -> List[str]:
    preferred = []
    for col in [
        "full_name", "full_name_norm",
        roles.get("full_name", ""),
        "first_name_norm", "last_name_norm",
        roles.get("first_name", ""), roles.get("last_name", "")
    ]:
        if col and col in columns:
            preferred.append(col)
    for col in [
        "address_norm", roles.get("address", ""),
        "city_norm", roles.get("city", ""),
        "state_norm", roles.get("state", ""),
        "zip_norm", roles.get("zip", "")
    ]:
        if col and col in columns:
            preferred.append(col)
    for col in ["email_norm", roles.get("email", ""), "phone_digits", roles.get("phone", "")]:
        if col and col in columns:
            preferred.append(col)
    if not preferred:
        sp = get_or_create_spark()
        preferred = [f.name for f in sp.createDataFrame([], struct=columns).schema if isinstance(f.dataType, StringType) and f.name not in ["unique_id", "cluster_id"]]
    seen = set()
    ordered = []
    for c in preferred:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

def _apply_enhancements_to_record_df(df: SparkDataFrame, roles: Dict[str, Any]) -> SparkDataFrame:
    df_out = df
    for role, col_name in roles.items():
        if col_name not in df.columns:
            continue
        if role in ["first_name", "last_name", "city", "address", "full_name"]:
            df_out = df_out.withColumn(f"{col_name}_norm", trim(lower(col(col_name))))
        if role in ["first_name", "last_name"]:
            df_out = df_out.withColumn(f"{col_name}_metaphone", expr(f"double_metaphone({col_name})"))
        if role == "email":
            df_out = df_out.withColumn(f"{col_name}_norm", trim(lower(col(col_name))))
        if role == "phone":
            df_out = df_out.withColumn(f"{col_name}_digits", regexp_replace(col(col_name), r"\D", ""))
        if role == "zip":
            df_out = df_out.withColumn(f"{col_name}_norm", trim(lower(col(col_name))))
    return df_out

# ----------------------------
# Duplicate Checking Logic
# ----------------------------
from datetime import datetime, date
from pyspark.sql.types import DateType, TimestampType, IntegerType, LongType, FloatType, DoubleType, BooleanType, StringType

def _check_record_against_clusters_fast(
    job: Dict[str, Any],
    record: Dict[str, Any],
    linker: Linker,
    roles: Dict[str, Any],
    df_enhanced: SparkDataFrame,
    report_df: SparkDataFrame,
    total_rows: int
) -> Dict[str, Any]:
    start_time = time.time()
    sp = get_or_create_spark()

    # 1. Prepare the new incoming record
    r = ensure_name_aliases(ensure_first_last_from_name(record))
    expected_schema = df_enhanced.schema
    all_cols = [f.name for f in expected_schema]
    
    # Fill missing columns with None and handle type conversions
    new_record_dict = {c: r.get(c) for c in all_cols}
    new_record_dict["unique_id"] = "new_record_to_check"
    
    # Convert values to match schema types
    for field in expected_schema:
        col_name = field.name
        value = new_record_dict.get(col_name)
        if value is None:
            continue
        try:
            if isinstance(field.dataType, StringType):
                new_record_dict[col_name] = str(value)
            elif isinstance(field.dataType, (IntegerType, LongType)):
                new_record_dict[col_name] = int(float(str(value))) if str(value).replace(".", "").replace("-", "").isdigit() else None
            elif isinstance(field.dataType, (FloatType, DoubleType)):
                new_record_dict[col_name] = float(value) if str(value).replace(".", "").replace("-", "").isdigit() else None
            elif isinstance(field.dataType, BooleanType):
                if isinstance(value, bool):
                    new_record_dict[col_name] = value
                elif isinstance(value, str):
                    normalized = value.lower()
                    new_record_dict[col_name] = True if normalized == "true" else False if normalized == "false" else None
                else:
                    new_record_dict[col_name] = None
            elif isinstance(field.dataType, DateType):
                if isinstance(value, str):
                    try:
                        # Try multiple date formats
                        date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y"]
                        parsed_date = None
                        for fmt in date_formats:
                            try:
                                parsed_date = datetime.strptime(value, fmt).date()
                                break
                            except ValueError:
                                continue
                        new_record_dict[col_name] = parsed_date if parsed_date else None
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not parse date for column {col_name}: {value}. Setting to None. Error: {e}")
                        new_record_dict[col_name] = None
                else:
                    new_record_dict[col_name] = None
            elif isinstance(field.dataType, TimestampType):
                if isinstance(value, str):
                    try:
                        # Try multiple timestamp formats
                        timestamp_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]
                        parsed_timestamp = None
                        for fmt in timestamp_formats:
                            try:
                                parsed_timestamp = datetime.strptime(value, fmt)
                                break
                            except ValueError:
                                continue
                        new_record_dict[col_name] = parsed_timestamp if parsed_timestamp else None
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not parse timestamp for column {col_name}: {value}. Setting to None. Error: {e}")
                        new_record_dict[col_name] = None
                else:
                    new_record_dict[col_name] = None
            else:
                # For unsupported types, keep as string or None
                new_record_dict[col_name] = str(value) if value is not None else None
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert value for column {col_name}: {value} to {field.dataType}. Setting to None. Error: {e}")
            new_record_dict[col_name] = None
    
    # Create and enhance the single-row DataFrame
    row_df_raw = sp.createDataFrame([new_record_dict], schema=expected_schema)
    row_df_enhanced = _apply_enhancements_to_record_df(row_df_raw, roles)
    
    print(f"Record preparation took {time.time() - start_time:.2f}s")

    # 2. Find potential matches using the robust Splink function
    # This replaces the brittle manual candidate selection.
    matches = linker.inference.find_matches_to_new_records(row_df_enhanced)
    df_matches = matches.as_spark_dataframe()
    
    print(f"Splink's find_matches_to_new_records took {time.time() - start_time:.2f}s")

    if df_matches.rdd.isEmpty():
        print(f"No potential matches found. Total time: {time.time() - start_time:.2f}s")
        return {"result": "unique", "cluster_id": "N/A", "partition_group": "N/A", "match_probability": 0.0}

    # 3. Fuzzy Re-scoring (Fine-tuning)
    # This logic remains powerful but is now applied to a better set of candidates.
    MAX_CANDIDATES_FOR_RESCORE = 2000
    df_matches_limited = df_matches.orderBy(col("match_probability").desc()).limit(MAX_CANDIDATES_FOR_RESCORE)
    match_rows = [r.asDict() for r in df_matches_limited.select("unique_id_l", "match_probability").collect()]

    if not match_rows:
        print(f"No matches after limiting. Total time: {time.time() - start_time:.2f}s")
        return {"result": "unique", "cluster_id": "N/A", "partition_group": "N/A", "match_probability": 0.0}

    candidate_ids_for_scoring = [r["unique_id_l"] for r in match_rows]
    
    # Fetch the full records for the best candidates to perform fuzzy matching
    candidate_rows_raw = df_enhanced.filter(col("unique_id").isin(candidate_ids_for_scoring)).limit(MAX_CANDIDATES_FOR_RESCORE).collect()
    candidate_map = {row["unique_id"]: row.asDict() for row in candidate_rows_raw}
    new_record_map = row_df_enhanced.limit(1).collect()[0].asDict()

    # Define columns for fuzzy scoring
    fuzzy_cols = []
    name_like = ["full_name", "first_name", "last_name", "surname"]
    other_roles = ["address", "city", "state"]
    for role_group in (name_like, other_roles):
        for role in role_group:
            c = roles.get(role)
            # Use the normalized column if it exists
            norm_col = f"{c}_norm" if c else None
            if norm_col and norm_col in all_cols:
                fuzzy_cols.append(norm_col)
    fuzzy_cols = list(set(fuzzy_cols))

    results = []
    for m in match_rows:
        uid = m["unique_id_l"]
        base_prob = float(m.get("match_probability", 0.0) or 0.0)
        cand = candidate_map.get(uid, {})
        
        fuzzy_score = 0.0
        if fuzzy_cols:
            sim_sum, weight_sum = 0.0, 0.0
            for col_name in fuzzy_cols:
                w = 2.0 if any(n in col_name for n in name_like) else 1.0
                weight_sum += w
                # Safely get values, defaulting to empty string
                l_val = cand.get(col_name, "") or ""
                r_val = new_record_map.get(col_name, "") or ""
                try:
                    sim = JaroWinkler.similarity(str(l_val), str(r_val)) * 100.0
                except Exception:
                    sim = 0.0
                sim_sum += w * sim
            fuzzy_score = (sim_sum / weight_sum) if weight_sum > 0 else 0.0
        
        boost = (fuzzy_score - 85.0) / 15.0 * 0.1 if fuzzy_score > 85.0 else 0.0
        adjusted_prob = min(1.0, base_prob + boost)
        results.append({"unique_id_l": uid, "adjusted_prob": float(adjusted_prob), "fuzzy_score": float(fuzzy_score)})

    if not results:
        print(f"Fuzzy scoring yielded no results. Total time: {time.time() - start_time:.2f}s")
        return {"result": "unique", "cluster_id": "N/A", "partition_group": "N/A", "match_probability": 0.0}

    best = max(results, key=lambda x: x["adjusted_prob"])
    max_prob = float(best["adjusted_prob"])
    best_uid_l = best["unique_id_l"]

    print(f"Fuzzy scoring finished at {time.time() - start_time:.2f}s")

    # 4. Final Classification
    adaptive_threshold = 0.99
    if total_rows < 1000: adaptive_threshold = 0.85
    elif total_rows < 5000: adaptive_threshold = 0.9

    if max_prob >= adaptive_threshold: result_type = "duplicate"
    elif max_prob >= 0.75: result_type = "potential_duplicate"
    else: result_type = "unique"

    if result_type == "unique":
        print(f"Classified as unique. Total time: {time.time() - start_time:.2f}s")
        return {"result": result_type, "cluster_id": "N/A", "partition_group": "N/A", "match_probability": float(max_prob)}

    # 5. Retrieve cluster information for the best match
    print(f"Looking up best match with unique_id: '{best_uid_l}' (type: {type(best_uid_l)})")
    
    match_info_rows = report_df.filter(
        col("unique_id").cast(StringType()) == lit(str(best_uid_l))
    ).select("cluster_id").collect()
    
    if not match_info_rows:
        return {
            "result": result_type,
            "cluster_id": "N/A (lookup failed)",
            "match_probability": float(max_prob),
        }

    match_info_row = match_info_rows[0]
    print(f"Final result retrieved. Total time: {time.time() - start_time:.2f}s")
    return {
        "result": result_type,
        "cluster_id": str(match_info_row["cluster_id"]),
        "match_probability": float(max_prob),
    }
#### SAVING TO MINIO

def _write_deduped_to_trino(parquet_dir: str, trino_cfg: Dict[str, Any], schema: str, table: str, job_id: str) -> None:
    """Upload report Parquet to MinIO, create temp external table, CTAS to managed deduped table if not exists, cleanup."""
    try:
        # Hardcoded MinIO creds from usethis.py (unchanged)
        S3_ENDPOINT = config.S3_ENDPOINT
        S3_ACCESS_KEY = config.S3_ACCESS_KEY
        S3_SECRET_KEY = config.S3_SECRET_KEY
        S3_BUCKET = config.S3_BUCKET
        S3_PREFIX = f"report/{job_id}/"  # Unique per job to avoid conflicts

        # Managed table name: {table}_deduped
        managed_table = f"{table}_deduped"
        temp_external_table = f"{managed_table}_tmp_external"
        source_path = f"s3a://{S3_BUCKET}/{S3_PREFIX}"

        # Step 1: Upload Parquet files to MinIO
        s3 = boto3.resource(
            's3',
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            config=Config(signature_version='s3v4')
        )
        bucket = s3.Bucket(S3_BUCKET)
        local_folder = Path(parquet_dir)
        for file_path in local_folder.glob("*.parquet"):
            key = f"{S3_PREFIX}{file_path.name}"
            bucket.upload_file(str(file_path), key)

        # Step 2: Infer schema from first Parquet file
        parquet_file = next(local_folder.glob("*.parquet"))
        pq_table = pq.read_table(parquet_file)
        pq_schema = pq_table.schema

        def pyarrow_to_trino(pa_type):
            pa_type = str(pa_type)
            if "string" in pa_type:
                return "VARCHAR"
            elif "int32" in pa_type:
                return "INTEGER"
            elif "int64" in pa_type:
                return "BIGINT"
            elif "float" in pa_type:
                return "REAL"
            elif "double" in pa_type:
                return "DOUBLE"
            elif "bool" in pa_type:
                return "BOOLEAN"
            elif "date" in pa_type:
                return "DATE"
            elif "timestamp" in pa_type:
                return "TIMESTAMP"
            else:
                return "VARCHAR"

        columns_sql = [f"{field.name} {pyarrow_to_trino(field.type)}" for field in pq_schema]
        columns_sql_str = ",\n    ".join(columns_sql)

        # Step 3: Connect to Trino
        conn = trino.dbapi.connect(
            host=trino_cfg.get("host", "localhost"),
            port=int(trino_cfg.get("port", 8080)),
            user=trino_cfg.get("user", config.TRINO_DEFAULT_USER),
            catalog=trino_cfg.get("catalog", "hive"),
            schema=schema
        )
        cursor = conn.cursor()

        # Step 4: Drop temp external if exists and create it
        cursor.execute(f"DROP TABLE IF EXISTS {schema}.{temp_external_table}")
        create_external_sql = f"""
        CREATE TABLE {schema}.{temp_external_table} (
            {columns_sql_str}
        )
        WITH (
            external_location = '{source_path}',
            format = 'PARQUET'
        )
        """
        cursor.execute(create_external_sql)

        # Step 5: Check if managed table exists
        table_exists = True
        try:
            cursor.execute(f"SELECT 1 FROM {schema}.{managed_table} LIMIT 1")
            cursor.fetchall()  # Consume result to avoid errors
        except trino.exceptions.TrinoQueryError:
            table_exists = False

        if table_exists:
            jobs[job_id]["warning"] = f"Deduped table '{managed_table}' already exists. Skipping creation."
        else:
            # Step 6: CTAS to managed table
            insert_sql = f"""
            CREATE TABLE {schema}.{managed_table} AS
            SELECT * FROM {schema}.{temp_external_table}
            """
            cursor.execute(insert_sql)

        # Step 7: Cleanup
        cursor.execute(f"DROP TABLE {schema}.{temp_external_table}")
        objects_to_delete = bucket.objects.filter(Prefix=S3_PREFIX)
        delete_requests = [{'Key': obj.key} for obj in objects_to_delete]
        if delete_requests:
            bucket.delete_objects(Delete={'Objects': delete_requests})

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Warning: Failed to write deduped table to Trino: {str(e)}")  # Log but don't fail job
# ----------------------------
# Job orchestration
# ----------------------------
def _update_progress(job_id: str, stage: str, pct: float) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    job["progress"] = max(0.0, min(1.0, pct))
    job["stage"] = stage
    job["updated_at"] = datetime.utcnow().isoformat()

# ----------------------------
# Job worker
# ----------------------------
def _run_dedupe_job(job_id: str, params: Dict[str, Any]) -> None:
    """
    Background worker to run Splink dedupe with singleton inclusion and safe LSH fallback.
    """
    from pyspark.sql.types import StringType
    try:
        _update_progress(job_id, "starting", 0.02)
        sp = get_or_create_spark()
        db_api = SparkAPI(spark_session=sp)

        # --- Data Loading and Validation ---
        if not connection.get("connected"):
            raise RuntimeError("Trino is not connected. Please POST to /connect first.")
        trino_cfg = connection.get("trino")
        if not trino_cfg:
            raise RuntimeError("Trino configuration missing. Did /connect succeed?")
        src = params.get("source") or {}
        table = src.get("table") or params.get("table")
        schema = params.get("schema")
        if schema and not trino_cfg.get("schema"):
            trino_cfg = {**trino_cfg, "schema": schema}
        if not table and schema:
            t = params.get("table")
            if t:
                table = f"{schema}.{t}"
        if not table:
            raise RuntimeError("No source table specified.")
        df_src = read_trino_table_as_spark_df(trino_cfg, table)
        if "unique_id" not in df_src.columns:
            df_src = df_src.withColumn("unique_id", monotonically_increasing_id().cast(StringType()))

        total_rows = df_src.count()
        jobs[job_id]["total_rows"] = total_rows

        rows_per_partition = 50000
        min_partitions = 16
        max_partitions = 200
        num_partitions = max(min_partitions, min(max_partitions, (total_rows + rows_per_partition - 1) // rows_per_partition))
        sp.conf.set("spark.sql.shuffle.partitions", str(num_partitions))
        print(f"Set spark.sql.shuffle.partitions to {num_partitions} based on total_rows={total_rows}")

        # --- Auto-Settings and DataFrame Caching ---
        _update_progress(job_id, "auto_blocking", 0.10)
        settings, roles, diagnostics, df_enhanced = ab.auto_generate_settings(
            df_src, db_api=db_api, spark=sp
        )
        print("Generated settings dict:", json.dumps(settings, indent=4))
#         #settings, roles, diagnostics, df_enhanced = ab.auto_generate_settings(
#     df_src, db_api=db_api, spark=sp, max_rules=5, max_comparisons=20_000_000, max_comparisons_cap=100_000_000
# )
        from pyspark.storagelevel import StorageLevel
        df_enhanced.persist(StorageLevel.MEMORY_AND_DISK)
        _ = df_enhanced.count()
        df_enhanced_path = os.path.join(OUTPUTS_DIR, f"df_enhanced_{job_id}.parquet")
        df_enhanced.write.mode("overwrite").parquet(df_enhanced_path)
        jobs[job_id]["df_enhanced_path"] = df_enhanced_path

        _update_progress(job_id, "linker_init", 0.22)
        training_df = df_enhanced.sample(0.4, seed=42).cache()
        _ = training_df.count()

        # --- Model Training ---
        training_linker = Linker(training_df, settings, db_api=db_api)
        deterministic_rules = [d.get("rule") for d in diagnostics if d.get("kept") and d.get("rule") is not None]

        _update_progress(job_id, "training_prob", 0.28)
        try:
            training_linker.training.estimate_probability_two_random_records_match(
                deterministic_matching_rules=deterministic_rules, recall=0.95
            )
        except Exception:
            training_linker.training.estimate_probability_two_random_records_match(
                deterministic_matching_rules=deterministic_rules, recall=1.0
            )

        _update_progress(job_id, "training_u", 0.32)
        training_linker.training.estimate_u_using_random_sampling(max_pairs=2e6)

        _update_progress(job_id, "training_em", 0.40)
        em_rule = deterministic_rules[0] if deterministic_rules else None
        if em_rule:
            training_linker.training.estimate_parameters_using_expectation_maximisation(em_rule)
        else:
            training_linker.training.estimate_parameters_using_expectation_maximisation()

        model_path = os.path.join(OUTPUTS_DIR, f"trained_model_{job_id}.json")
        training_linker.misc.save_model_to_json(model_path, overwrite=True)
        training_df.unpersist()

        # --- Prediction and Clustering ---
        _update_progress(job_id, "predict", 0.55)
        with io.open(model_path, "r", encoding="utf-8") as f:
            trained_settings = json.load(f)
        inference_linker = Linker(df_enhanced, trained_settings, db_api=db_api)
        predictions_splink = inference_linker.inference.predict()

        threshold = 0.99
        if total_rows < 1000:
            threshold = 0.85
        elif total_rows < 5000:
            threshold = 0.9
        threshold = float(params.get("threshold", threshold))

        _update_progress(job_id, "cluster", 0.70)
        clusters_splink = inference_linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions_splink, threshold_match_probability=threshold
        )

        # --- Include singletons in final clusters ---
        df_predictions = predictions_splink.as_spark_dataframe()

        # ✅ Deduplicate Splink cluster output first
        df_clusters_initial = clusters_splink.as_spark_dataframe().dropDuplicates(["unique_id"])

        from pyspark.sql.functions import col, coalesce, lit, concat_ws
        from pyspark.sql.types import StringType

        df_clusters_initial = df_clusters_initial.alias("preds")
        df_enhanced_aliased = df_enhanced.alias("enh")
        join_condition = col("preds.unique_id") == col("enh.unique_id")
        enhanced_cols = [col(f"enh.{c_name}") for c_name in df_enhanced_aliased.columns]

        # ✅ Join + drop duplicate unique_ids (ensures row count = input)
        full_clusters_sdf = df_enhanced_aliased.join(
            df_clusters_initial,
            join_condition,
            "left_outer"
        ).select(
            coalesce(
                col("preds.cluster_id").cast(StringType()),
                concat_ws("", lit("singleton_"), col("enh.unique_id").cast(StringType()))
            ).alias("cluster_id"),
            *enhanced_cols
        ).dropDuplicates(["unique_id"]).persist(StorageLevel.MEMORY_AND_DISK)

        print(f"✅ Splink clusters prepared: {full_clusters_sdf.count()} rows (should match input).")


        from concurrent.futures import ThreadPoolExecutor
        full_clusters_sdf = full_clusters_sdf.persist(StorageLevel.MEMORY_AND_DISK)
        _ = full_clusters_sdf.count()

        clusters_path = os.path.join(OUTPUTS_DIR, f"splink_clusters_{job_id}.csv")
        write_single_csv(full_clusters_sdf, clusters_path, sort_cols=["cluster_id", "unique_id"])
        print(f"✅ Cluster results written to: {clusters_path}")

        if "cluster_id" in full_clusters_sdf.columns and "unique_id" in full_clusters_sdf.columns:
            ordered = ["cluster_id", "unique_id"] + [
                c for c in full_clusters_sdf.columns if c not in ["cluster_id", "unique_id"]
            ]
            full_clusters_sdf = full_clusters_sdf.select(*ordered)

        # --- Simplified final clustering (skip GraphFrames; keep 1:1 record count) ---
        _update_progress(job_id, "advanced_clustering", 0.75)

        roles_path = os.path.join(OUTPUTS_DIR, f"roles_{job_id}.json")
        with io.open(roles_path, "w", encoding="utf-8") as f:
            json.dump(roles, f)
        jobs[job_id]["roles_path"] = roles_path

        # ✅ Assign partition_group = cluster_id (clean, consistent)
        from pyspark.sql.functions import col
        full_clusters_sdf = full_clusters_sdf.withColumn("partition_group", col("cluster_id").cast("string"))

        # ✅ Make sure final record count = input
        full_clusters_sdf = full_clusters_sdf.dropDuplicates(["unique_id"]).persist(StorageLevel.MEMORY_AND_DISK)
        print(f"✅ Final report row count: {full_clusters_sdf.count()} (should match input).")

        report_path = os.path.join(OUTPUTS_DIR, f"reports_{job_id}.csv")
        report_parquet_path = os.path.join(OUTPUTS_DIR, f"reports_{job_id}.parquet")

        def _write_csv():
            final_count = full_clusters_sdf.count()
            if final_count != total_rows:
                raise RuntimeError(f"Final CSV would have {final_count} rows, expected {total_rows}")
            sort_order = ["partition_group", "cluster_id", "unique_id"]
            write_single_csv(full_clusters_sdf, report_path, sort_cols=sort_order)

        def _write_parquet():
            full_clusters_sdf.write.mode("overwrite").parquet(report_parquet_path)

        with ThreadPoolExecutor(max_workers=10) as ex:
            ex.submit(_write_csv)
            ex.submit(_write_parquet)

        jobs[job_id]["report_parquet_path"] = report_parquet_path
        preds_path = os.path.join(OUTPUTS_DIR, f"splink_predictions_{job_id}.csv")
        write_single_csv(df_predictions, preds_path)
        _write_deduped_to_trino(report_parquet_path, connection["trino"], schema, table, job_id)
        _update_progress(job_id, "profile", 0.93)
        # profile_html_path = os.path.join(OUTPUTS_DIR, f"profile_{job_id}.html")
        # try:
        #     from splink.internals.profile_data import profile_columns
        #     prof_cols = [f.name for f in df_src.schema][:25]
        #     linker_for_profile = Linker(df_src, settings, db_api=db_api)
        #     html = profile_columns(linker_for_profile, prof_cols)
        #     with io.open(profile_html_path, "w", encoding="utf-8") as f:
        #         f.write(html if isinstance(html, str) else str(html))
        # except Exception:
        #     with io.open(profile_html_path, "w", encoding="utf-8") as f:
        #         f.write("<html><body><p>Profile unavailable.</p></body></html>")

        df_enhanced.unpersist()

        with cache_lock:
            sp = get_or_create_spark()
            db_api = SparkAPI(spark_session=sp)
            with io.open(model_path, "r", encoding="utf-8") as f:
                trained_settings = json.load(f)
            df_enhanced_loaded = sp.read.parquet(df_enhanced_path)
            df_enhanced_loaded.persist(StorageLevel.MEMORY_AND_DISK)
            _ = df_enhanced_loaded.count()
            cached_df_enhanced[job_id] = df_enhanced_loaded
            linker = Linker(df_enhanced_loaded, trained_settings, db_api=db_api)
            cached_linkers[job_id] = linker
            report_df_loaded = sp.read.parquet(report_parquet_path)
            report_df_loaded.persist(StorageLevel.MEMORY_AND_DISK)
            _ = report_df_loaded.count()
            cached_report_df[job_id] = report_df_loaded
            with io.open(roles_path, "r", encoding="utf-8") as f:
                cached_roles[job_id] = json.load(f)
            cached_total_rows[job_id] = total_rows

        jobs[job_id].update({
            "status": "completed",
            "progress": 1.0,
            "stage": "done",
            "outputs": {
                "predictions": preds_path,
                "clusters": clusters_path,
                "report": report_path,
                "model": model_path,
                #"profile_html": profile_html_path,
            },
            "completed_at": datetime.utcnow().isoformat()
        })

    except Exception as e:
        err_txt = traceback.format_exc()
        error_path = os.path.join(OUTPUTS_DIR, f"error_{job_id}.txt")
        try:
            with io.open(error_path, "w", encoding="utf-8") as f:
                f.write(err_txt)
        except Exception:
            error_path = None
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["error_file"] = error_path
        jobs[job_id]["stage"] = "error"
        jobs[job_id]["progress"] = 1.0

# ----------------------------
# Routes
# ----------------------------
def _default_ui_config() -> Dict[str, Any]:
    return {
        "TRINO_HOST": os.environ.get("TRINO_HOST", ""),
        "TRINO_PORT": os.environ.get("TRINO_PORT", ""),
        "TRINO_USER": os.environ.get("TRINO_USER", ""),
        "TRINO_CATALOG": os.environ.get("TRINO_CATALOG", ""),
    }

@app.route("/")
def index():
    return render_template("index.html", defaults=_default_ui_config())

@app.route("/session", methods=["GET"])
def get_session():
    return jsonify({
        "ok": True,
        "session_id": SESSION_ID,
        "connected": connection.get("connected", False),
        "trino": connection.get("trino"),
    })

@app.route("/connect", methods=["POST"])
def connect_trino():
    data = request.get_json(force=True)
    required = ["host", "port", "catalog", "user"]
    for k in required:
        if k not in data or data[k] in (None, ""):
            return jsonify({"ok": False, "error": f"Missing '{k}'"}), 400

    cfg = {
        "host": data["host"],
        "port": data["port"],
        "catalog": data["catalog"],
        "schema": data.get("schema"),
        "user": data["user"],
    }

    try:
        with _get_trino_connection({**cfg, "schema": None}) as conn:
            cur = conn.cursor()
            cur.execute("SHOW SCHEMAS")
            schemas = [row[0] for row in cur.fetchall()]
    except Exception as e:
        return jsonify({"ok": False, "error": f"Failed to connect to Trino: {str(e)}"}), 400

    connection["connected"] = True
    connection["trino"] = cfg
    get_or_create_spark()
    return jsonify({"ok": True, "schemas": schemas})

@app.route("/tables", methods=["POST"])
def list_tables():
    if not connection.get("connected"):
        return jsonify({"ok": False, "error": "Not connected"}), 400
    cfg = connection["trino"].copy()
    data = request.get_json(force=True) if request.data else {}
    schema = data.get("schema") or cfg.get("schema")
    if not schema:
        return jsonify({"ok": False, "error": "Schema is required"}), 400
    try:
        with _get_trino_connection({**cfg, "schema": schema}) as conn:
            cur = conn.cursor()
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
        return jsonify({"ok": True, "tables": tables})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/columns", methods=["POST"])
def list_columns():
    if not connection.get("connected"):
        return jsonify({"ok": False, "error": "Not connected"}), 400
    cfg = connection["trino"]
    data = request.get_json(force=True) if request.data else {}
    table = data.get("table")
    schema = data.get("schema") or cfg.get("schema")
    if not table:
        return jsonify({"ok": False, "error": "Missing table"}), 400

    try:
        parts = table.split(".")
        if len(parts) == 1:
            schema, tname = schema, parts[0]
        elif len(parts) == 2:
            schema, tname = parts[0], parts[1]
        else:
            _, schema, tname = parts[-3], parts[-2], parts[-1]
        with _get_trino_connection({**cfg, "schema": schema}) as conn:
            cur = conn.cursor()
            cur.execute(f"DESCRIBE {tname}")
            rows = [{"name": r[0], "type": r[1]} for r in cur.fetchall()]
        return jsonify({"ok": True, "columns": rows})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/reset", methods=["POST"])
def reset_state():
    global jobs, current_session_id, cached_linkers, cached_roles, cached_df_enhanced, cached_report_df, cached_total_rows
    jobs = {}
    current_session_id = uuid.uuid4().hex[:16]
    with cache_lock:
        for job_id in cached_df_enhanced:
            try:
                cached_df_enhanced[job_id].unpersist()
            except Exception: pass
        for job_id in cached_report_df:
            try:
                cached_report_df[job_id].unpersist()
            except Exception: pass
        cached_linkers.clear()
        cached_roles.clear()
        cached_df_enhanced.clear()
        cached_report_df.clear()
        cached_total_rows.clear()
    _purge_outputs_dir()
    return jsonify({"ok": True, "session_id": current_session_id})

@app.route("/run", methods=["POST"])
def run_job():
    if not connection.get("connected"):
        return jsonify({"error": "Not connected"}), 400

    params = request.get_json(force=True) if request.data else {}
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "id": job_id,
        "status": "running",
        "progress": 0.0,
        "stage": "created",
        "created_at": datetime.utcnow().isoformat(),
        "outputs": {},
        "params": params,
    }

    t = threading.Thread(target=_run_dedupe_job, args=(job_id, params), daemon=True)
    t.start()

    return jsonify({"ok": True, "job_id": job_id})

@app.route("/progress/<job_id>")
def progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    p = int(round((job.get("progress") or 0) * 100))
    status = job.get("status")
    if status == "failed":
        status_out = "error"
    elif status == "completed":
        status_out = "completed"
    else:
        status_out = "running"
    return jsonify({
        "ok": True,
        "id": job_id,
        "status": status_out,
        "progress": p,
        "stage": job.get("stage"),
        "outputs": job.get("outputs", {}),
        "error": job.get("error"),
        "error_file": job.get("error_file"),
        "warning": job.get("warning")
    })

@app.route("/download/<job_id>")
def download_clusters(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") not in ("completed", "failed"):
        return jsonify({"ok": False, "error": "job not completed"}), 400
    path = job.get("outputs", {}).get("clusters")
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "clusters not found"}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route("/report/<job_id>")
def download_report(job_id: str):
    job = jobs.get(job_id)
    if not job or job.get("status") not in ("completed", "failed"):
        return jsonify({"ok": False, "error": "job not completed"}), 400
    path = job.get("outputs", {}).get("report")
    if not path or not os.path.exists(path):
        return jsonify({"ok": False, "error": "report not found"}), 404
    return send_file(path, as_attachment=True, download_name=os.path.basename(path))

@app.route("/check_record", methods=["POST"])
def check_record():
    start_time = time.time()
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    record = data.get("record") or {}
    if not job_id or job_id not in jobs:
        return jsonify({"ok": False, "error": "invalid job_id"}), 400
    job = jobs[job_id]
    
    if job.get("status") != "completed":
        return jsonify({"ok": False, "error": f"Job '{job_id}' is not completed. Current status: {job.get('status')}"}), 400
    
    model_path = job.get("outputs", {}).get("model")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"ok": False, "error": "trained model not available for this job"}), 400

    with cache_lock:
        # Check if essential data is cached. If not, load it.
        if job_id not in cached_linkers:
            sp = get_or_create_spark()
            df_enhanced_path = job.get("df_enhanced_path")
            report_parquet_path = job.get("report_parquet_path")
            roles_path = job.get("roles_path")
            if not all([df_enhanced_path, report_parquet_path, roles_path]):
                return jsonify({"ok": False, "error": "data not available for checking"}), 400
            
            with io.open(model_path, "r", encoding="utf-8") as f:
                model_settings = json.load(f)
            
            df_enhanced = sp.read.parquet(df_enhanced_path).persist(StorageLevel.MEMORY_AND_DISK)
            _ = df_enhanced.count()
            cached_df_enhanced[job_id] = df_enhanced
            
            db_api = SparkAPI(spark_session=sp)
            linker = Linker(df_enhanced, model_settings, db_api=db_api)
            cached_linkers[job_id] = linker
            
            report_df = sp.read.parquet(report_parquet_path).persist(StorageLevel.MEMORY_AND_DISK)
            _ = report_df.count()
            cached_report_df[job_id] = report_df
            
            with io.open(roles_path, "r", encoding="utf-8") as f:
                cached_roles[job_id] = json.load(f)
            
            cached_total_rows[job_id] = job.get("total_rows", df_enhanced.count())
        
        # Retrieve data from cache
        linker = cached_linkers[job_id]
        roles = cached_roles[job_id]
        df_enhanced = cached_df_enhanced[job_id]
        report_df = cached_report_df[job_id]
        total_rows = cached_total_rows[job_id]

    result = _check_record_against_clusters_fast(job, record, linker, roles, df_enhanced, report_df, total_rows)
    print(f"Total route time: {time.time() - start_time:.2f}s")
    return jsonify({
        "ok": True,
        **result
    })

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    traceback.print_exc()
    return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/favicon.ico')
def favicon():
    # Returns an empty 204 No Content response
    return '', 204

if __name__ == "__main__":
    get_or_create_spark()
    app.run(host="0.0.0.0", port=5000, debug=True)