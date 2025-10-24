from __future__ import annotations
import re
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import (
    col, lower, upper, trim, regexp_replace, substring, split,
    element_at, expr, when, length, sha2, lit, concat, regexp_extract
)
from pyspark.sql.types import StringType, StructField, StructType
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import to_date, to_timestamp, lit
from pyspark.sql import functions as F

# Splink imports
import splink.comparison_library as cl
from splink import SettingsCreator, block_on, SparkAPI
from splink.blocking_analysis import count_comparisons_from_blocking_rule

# --- helper functions ---

def _nonnull_share(df: SparkDataFrame, col_name: str, total_rows: int) -> float:
    """Calculate the share of non-null values in a Series."""
    if total_rows == 0:
        return 0.0
    return df.filter(col(col_name).isNotNull()).count() / total_rows

def _cardinality_ratio(df: SparkDataFrame, col_name: str, total_rows: int) -> float:
    """Calculate the ratio of unique values to total values in a Series."""
    if total_rows == 0:
        return 0.0
    unique_count = df.select(col_name).distinct().count()
    return unique_count / total_rows

# --- role inference / derived columns / rule generation ---

def create_enhanced_column_mapper() -> Dict[str, List[str]]:
    """Expanded dictionary mapping roles to common column name aliases."""
    return {
        "first_name": ["first_name", "firstname", "given_name", "fname", "forename"],
        "last_name": ["last_name", "lastname", "surname", "lname", "family_name"],
        "full_name": ["full_name", "customer_name", "name", "person_name", "complete_name"],
        "email": ["email", "email_address", "e_mail", "contact_email"],
        "phone": ["phone", "mobile", "contact_no", "phone_number", "telephone"],
        "zip": ["zip", "zipcode", "postal_code", "pincode", "postcode"],
        "city": ["city", "town", "municipality"],
        "state": ["state", "province", "region", "county"],
        "address": ["address", "street", "street_address", "addr", "location"],
        "date": ["date", "dob", "birth_date", "order_date", "timestamp", "created_at"],
        "url": ["url", "website", "web_address"],
        "geo_lat": ["lat", "latitude", "geo_lat"],
        "geo_lon": ["lon", "long", "longitude", "geo_lon"],
        "currency": ["price", "amount", "cost", "revenue", "salary"],
        "numeric_id": ["id", "key", "account_number", "record_id", "customer_id", "user_id", "person_id", "orderkey"],
    }

def create_semantic_patterns() -> Dict[str, List[str]]:
    """Expanded dictionary of regex patterns to infer column roles."""
    return {
        "email": [r".*mail.*", r".*@.*"],
        "phone": [
            r".*(phone|mobile|tel|contact).*",
            r"^\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3,4}$",  # +1-123-456-7890, (123) 456-7890
            r"^\d{10}$",  # 1234567890
            r"^\(\d{3}\)\s*\d{3}-\d{4}$"  # (123) 456-7890
        ],
        "zip": [r".*(zip|postal|pin).*code.*"],
        "date": [
            r".*date.*", 
            r".*_at$", 
            r".*timestamp.*", 
            r"^\d{4}[/-]\d{2}[/-]\d{2}$",  # YYYY-MM-DD
            r"^\d{2}[/-]\d{2}[/-]\d{4}$",  # DD/MM/YYYY or MM-DD-YYYY
            r"^\d{2}[/-]\d{2}[/-]\d{2}$"  # DD/MM/YY or MM-DD-YY
        ],
        "timestamp": [
            r".*timestamp.*", 
            r".*_at$", 
            r"^\d{4}[/-]\d{2}[/-]\d{2}\s+\d{2}:\d{2}(:\d{2})?$",  # YYYY-MM-DD HH:MM(:SS)
            r"^\d{2}[/-]\d{2}[/-]\d{4}\s+\d{2}:\d{2}(:\d{2})?$",  # DD/MM/YYYY HH:MM(:SS)
            r"^\d{4}[/-]\d{2}[/-]\d{2}T\d{2}:\d{2}:\d{2}$"  # YYYY-MM-DDTHH:MM:SS
        ],
        "url": [
            r".*(url|website|http|www).*",
            r"^https?://.*",  # http://example.com
            r"^www\..*"  # www.example.com
        ],
        "geo_lat": [
            r".*lat(itude)?.*",
            r"^-?\d{1,3}\.\d+$",  # 40.7128
            r"^-?\d{1,3}°\d{1,2}'\d{1,2}(\.\d+)?\"[NS]$"  # 40°42'46.08"N
        ],
        "geo_lon": [
            r".*lon(gitude)?.*",
            r"^-?\d{1,3}\.\d+$",  # -74.0060
            r"^-?\d{1,3}°\d{1,2}'\d{1,2}(\.\d+)?\"[EW]$"  # 74°00'21.6"W
        ],
        "numeric_id": [r".*(id|key|number|no)$"],
        "name": [r".*name.*"],
        "address": [
            r".*(addr|address|street).*",
            r"^\d+\s+[A-Za-z\s]+(St|Ave|Rd|Blvd|Ln|Dr|Ct|Cir|Pl|Way).*$",  # 123 Main St
            r"^\d+\s*[-,]\s*[A-Za-z\s]+(Apt|Unit|Suite).*$"  # 456 Oak Rd, Apt 4B
        ],
        "geo": [r".*(city|state|province).*"]
    }

def infer_roles_enhanced(input_df: SparkDataFrame, sample_size: int = 10000, verbosity: bool = True) -> Dict[str, str]:
    """
    Infer column roles using a multi-pass approach: direct aliasing, regex patterns,
    and data type/distribution analysis.
    Uses a sample for efficiency on large datasets.
    """
    if input_df.isEmpty():
        return {}

    total_rows = input_df.count()
    if total_rows == 0:
        return {}
        
    lowercase_to_original = {c.lower(): c for c in input_df.columns}
    roles: Dict[str, str] = {}
    
    # Take a sample for efficient data-based inference
    sample_fraction = min(1.0, sample_size / total_rows) if total_rows > 0 else 0.0
    sample_df = input_df.sample(False, sample_fraction, seed=42) if total_rows > sample_size else input_df
    sample_df.persist(StorageLevel.MEMORY_AND_DISK)
    sample_df_count = sample_df.count()
    
    # Date/time formats
    date_formats = [
        "yyyy-MM-dd", "dd/MM/yyyy", "MM-dd-yyyy", "dd-MM-yy", "yyyy/MM/dd",
        "MM/dd/yy", "yyyyMMdd", "ddMMyyyy"
    ]
    timestamp_formats = [
        "yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd HH:mm", "dd/MM/yyyy HH:mm:ss",
        "MM-dd-yyyy HH:mm:ss", "yyyyMMdd HHmmss", "yyyy-MM-dd'T'HH:mm:ss"
    ]
    
    # Patterns for validation
    phone_patterns = [
        r"^\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3,4}$",
        r"^\(\d{3}\)\s*\d{3}-\d{4}$",
        r"^\d{10}$"
    ]
    address_patterns = [
        r"^\d+\s+[A-Za-z\s]+(St|Ave|Rd|Blvd|Ln|Dr|Ct|Cir|Pl|Way).*$",
        r"^\d+\s*[-,]\s*[A-Za-z\s]+(Apt|Unit|Suite).*$"
    ]
    url_patterns = [r"^https?://.*", r"^www\..*"]
    geo_patterns = [r"^-?\d{1,3}\.\d+$", r"^-?\d{1,3}°\d{1,2}'\d{1,2}(\.\d+)?\"[NSEW]$"]
    
    # Pass 1: Direct alias matching
    for role, aliases in create_enhanced_column_mapper().items():
        if role in roles: continue
        for alias in aliases:
            if alias in lowercase_to_original:
                col_original = lowercase_to_original[alias]
                if "unique_id" not in col_original.lower() and col_original not in roles.values():
                    roles[role] = col_original
                    break

    # Pass 2: Pattern-based matching (using column names)
    for role, patterns in create_semantic_patterns().items():
        if role in roles: continue
        for pattern in patterns:
            for col_lower, col_original in lowercase_to_original.items():
                if re.search(pattern, col_lower, re.IGNORECASE) and \
                   "unique_id" not in col_lower and \
                   col_original not in roles.values():
                    roles[role] = col_original
                    break
            if role in roles: break

    # Pass 3: Data type and distribution-based inference
    for col_name in input_df.columns:
        if col_name in roles.values() or "unique_id" in col_name.lower():
            continue
        
        non_null_share = _nonnull_share(input_df, col_name, total_rows)
        if non_null_share < 0.1:
            continue

        spark_dtype = input_df.schema[col_name].dataType
        
        if str(spark_dtype).startswith(("Integer", "Long", "Float", "Double")):
            if "numeric_id" not in roles:
                card_ratio = _cardinality_ratio(sample_df, col_name, sample_df_count)
                if card_ratio >= 0.85 and non_null_share >= 0.9:
                    roles["numeric_id"] = col_name
            elif "geo_lat" not in roles or "geo_lon" not in roles:
                # Check if values are in valid lat/lon range
                valid_geo = sample_df.select(
                    col(col_name).cast("double").alias("val")
                ).filter((col("val") >= -180) & (col("val") <= 180)).count()
                if valid_geo / sample_df_count > 0.5:
                    if col_name.lower().find("lat") != -1 and "geo_lat" not in roles:
                        roles["geo_lat"] = col_name
                    elif col_name.lower().find("lon") != -1 and "geo_lon" not in roles:
                        roles["geo_lon"] = col_name
        
        elif str(spark_dtype).startswith(("TimestampType", "DateType")) and ("date" not in roles or "timestamp" not in roles):
            if "timestamp" not in roles:
                roles["timestamp"] = col_name
            elif "date" not in roles:
                roles["date"] = col_name
        
        elif str(spark_dtype).startswith("StringType"):
            # Date/timestamp parsing
            is_date = False
            is_timestamp = False
            for fmt in date_formats:
                parsed = sample_df.select(
                    to_date(col(col_name).cast("string"), fmt).alias("parsed")
                ).filter(col("parsed").isNotNull()).count()
                if parsed / sample_df_count > 0.5:
                    is_date = True
                    break
            for fmt in timestamp_formats:
                parsed = sample_df.select(
                    to_timestamp(col(col_name).cast("string"), fmt).alias("parsed")
                ).filter(col("parsed").isNotNull()).count()
                if parsed / sample_df_count > 0.5:
                    is_timestamp = True
                    break
            if is_date and "date" not in roles:
                roles["date"] = col_name
            elif is_timestamp and "timestamp" not in roles:
                roles["timestamp"] = col_name
            else:
                # Phone validation
                if "phone" not in roles:
                    for pattern in phone_patterns:
                        valid = sample_df.select(
                            col(col_name).cast("string").alias("val")
                        ).filter(col("val").rlike(pattern)).count()
                        if valid / sample_df_count > 0.5:
                            roles["phone"] = col_name
                            break
                # Address validation
                elif "address" not in roles:
                    for pattern in address_patterns:
                        valid = sample_df.select(
                            col(col_name).cast("string").alias("val")
                        ).filter(col("val").rlike(pattern)).count()
                        if valid / sample_df_count > 0.5:
                            roles["address"] = col_name
                            break
                # URL validation
                elif "url" not in roles:
                    for pattern in url_patterns:
                        valid = sample_df.select(
                            col(col_name).cast("string").alias("val")
                        ).filter(col("val").rlike(pattern)).count()
                        if valid / sample_df_count > 0.5:
                            roles["url"] = col_name
                            break
                # Fallback: category or text
                else:
                    card_ratio = _cardinality_ratio(sample_df, col_name, sample_df_count)
                    if card_ratio < 0.1 and card_ratio > 0 and "category_enum" not in roles and non_null_share > 0.2:
                        roles["category_enum"] = col_name
                    elif card_ratio > 0.5 and "text_freeform" not in roles and non_null_share > 0.2:
                        roles["text_freeform"] = col_name
                
    sample_df.unpersist()
    return roles

def ensure_derived_columns_enhanced(input_df: SparkDataFrame, roles: Dict[str, str], total_rows: int, verbosity: bool = True) -> SparkDataFrame:
    """
    Safely create derived columns for normalization, phonetic matching, hashing, etc.
    Uses Java UDFs from the custom JAR for stability and performance.
    """
    from pyspark.sql.functions import to_date, to_timestamp, format_number

    df_transformed = input_df

    if total_rows == 0:
        return df_transformed

    # Date/time formats
    date_formats = [
        "yyyy-MM-dd", "dd/MM/yyyy", "MM-dd-yyyy", "dd-MM-yy", "yyyy/MM/dd",
        "MM/dd/yy", "yyyyMMdd", "ddMMyyyy"
    ]
    timestamp_formats = [
        "yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd HH:mm", "dd/MM/yyyy HH:mm:ss",
        "MM-dd-yyyy HH:mm:ss", "yyyyMMdd HHmmss", "yyyy-MM-dd'T'HH:mm:ss"
    ]

    if "full_name" in roles and roles["full_name"] in df_transformed.columns:
        full_name_col = roles["full_name"]
        if _nonnull_share(df_transformed, full_name_col, total_rows) > 0.2:
            df_transformed = df_transformed.withColumn(
                "temp_name_parts", split(when(col(full_name_col).isNull(), lit("")).otherwise(col(full_name_col).cast("string")), " ", 2)
            )
            if "first_name" not in roles:
                df_transformed = df_transformed.withColumn(
                    "first_name_derived", element_at(col("temp_name_parts"), 1)
                )
                roles["first_name"] = "first_name_derived"
            if "last_name" not in roles:
                df_transformed = df_transformed.withColumn(
                    "last_name_derived", element_at(col("temp_name_parts"), 2)
                )
                roles["last_name"] = "last_name_derived"
            df_transformed = df_transformed.drop("temp_name_parts")

    for name_type in ["first_name", "last_name"]:
        if name_type in roles and roles[name_type] in df_transformed.columns:
            base_col = roles[name_type]
            if _nonnull_share(df_transformed, base_col, total_rows) > 0.2:
                try:
                    df_transformed = df_transformed.withColumn(
                        f"{name_type}_norm",
                        lower(trim(expr(f"accent_remove(CAST(`{base_col}` AS STRING))")))
                    )
                    df_transformed = df_transformed.withColumn(
                        f"{name_type}_metaphone",
                        expr(f"double_metaphone({name_type}_norm)")
                    )
                except Exception as e:
                    if verbosity:
                        print(f"Warning: Could not create derived name columns for '{name_type}': {e}")

    for role, col_name in roles.items():
        if col_name not in df_transformed.columns or _nonnull_share(df_transformed, col_name, total_rows) <= 0.2:
            continue
        
        try:
            if role == "email":
                df_transformed = df_transformed.withColumn(
                    "email_norm", lower(trim(when(col(col_name).isNull(), lit("")).otherwise(col(col_name).cast("string"))))
                )
            elif role == "phone":
                # Normalize to digits only, extract last 7 digits
                df_transformed = df_transformed.withColumn(
                    "phone_digits",
                    regexp_replace(when(col(col_name).isNull(), lit("")).otherwise(col(col_name).cast("string")), r"[^\d]", "")
                )
                df_transformed = df_transformed.withColumn(
                    "phone_last7",
                    substring(col("phone_digits"), -7, 7)
                )
            elif role == "zip":
                df_transformed = df_transformed.withColumn(
                    "zip_norm", upper(regexp_replace(when(col(col_name).isNull(), lit("")).otherwise(col(col_name).cast("string")), r"\s", ""))
                )
            elif role == "address":
                # Normalize address: remove punctuation, lowercase, extract street
                addr_col = lower(trim(regexp_replace(
                    when(col(col_name).isNull(), lit("")).otherwise(col(col_name).cast("string")),
                    r"[,\.;#]", ""
                )))
                df_transformed = df_transformed.withColumn(
                    "address_norm", addr_col
                )
                df_transformed = df_transformed.withColumn(
                    "address_street",
                    regexp_extract(addr_col, r"^\d+\s+([a-z\s]+)(st|ave|rd|blvd|ln|dr|ct|cir|pl|way)", 1)
                )
            elif role in ("city", "state"):
                df_transformed = df_transformed.withColumn(
                    f"{role}_norm", lower(trim(expr(f"accent_remove(CAST(`{col_name}` AS STRING))")))
                )
            elif role == "date":
                # Safely normalize date fields (e.g., dob, birth_date)
                clean_date_col = col(col_name).cast("string")
                parsed_date = None
                for fmt in date_formats:
                    candidate = to_date(clean_date_col, fmt)
                    parsed_date = candidate if parsed_date is None else when(candidate.isNotNull(), candidate).otherwise(parsed_date)
                df_transformed = df_transformed.withColumn(
                    "date_norm",
                    when(parsed_date.isNotNull(), F.date_format(parsed_date, "yyyy-MM-dd")).otherwise(None)
                )

            elif role == "timestamp":
                ts_col = when(col(col_name).isNull(), lit("")).otherwise(col(col_name).cast("string"))
                for i, fmt in enumerate(timestamp_formats):
                    ts_col = when(
                        to_timestamp(col(col_name).cast("string"), fmt).isNotNull(),
                        to_timestamp(col(col_name).cast("string"), fmt)
                    ).otherwise(ts_col)
                df_transformed = df_transformed.withColumn(
                    "timestamp_norm",
                    when(ts_col.isNotNull(), expr("date_format(CAST(ts_col AS TIMESTAMP), 'yyyy-MM-dd HH:mm:ss')")).otherwise(None)
                )
            elif role == "url":
                url_col = lower(when(col(col_name).isNull(), lit("")).otherwise(col(col_name).cast("string")))
                df_transformed = df_transformed.withColumn(
                    "url_domain",
                    regexp_extract(url_col, r"https?://(?:www\.)?([^/]+)", 1)
                )
                df_transformed = df_transformed.withColumn(
                    "url_subdomain",
                    regexp_extract(url_col, r"https?://(?:www\.)?([a-z0-9-]+\.[a-z0-9-]+)", 1)
                )
            elif role in ("geo_lat", "geo_lon"):
                # Normalize to 6 decimal places
                df_transformed = df_transformed.withColumn(
                    f"{role}_norm",
                    format_number(col(col_name).cast("double"), 6)
                )
                # Round to 2 decimals for proximity
                df_transformed = df_transformed.withColumn(
                    f"{role}_rounded",
                    format_number(col(col_name).cast("double"), 2)
                )
            elif role == "numeric_id":
                df_transformed = df_transformed.withColumn(
                    f"{col_name}_hash",
                    sha2(when(col(col_name).isNull(), lit("")).otherwise(col(col_name).cast("string")), 256)
                )
        except Exception as e:
            if verbosity:
                print(f"Warning: Could not create derived column for role '{role}' on column '{col_name}': {e}")
            
    if "first_name" in roles and roles["first_name"] in df_transformed.columns:
        if _nonnull_share(df_transformed, roles["first_name"], total_rows) > 0.2:
            df_transformed = df_transformed.withColumn(
                "first_name_first_char",
                substring(when(col(roles["first_name"]).isNull(), lit("")).otherwise(col(roles["first_name"]).cast("string")), 1, 1).cast(StringType())
            )

    df_transformed.persist(StorageLevel.MEMORY_AND_DISK)
    return df_transformed

def generate_robust_blocking_rules(df: SparkDataFrame, roles: Dict[str, str], total_rows: int) -> Tuple[List[Tuple[str, object]], SparkDataFrame]:
    """
    Generate a diverse list of candidate blocking rules based on available columns.
    """
    rules = []

    def add_rule_if_valid(name, *cols):
        min_non_null_share = 0.3
        valid = True
        for c in cols:
            # If it's a Column expression (not a string), skip existence check
            if isinstance(c, str):
                if c not in df.columns or _nonnull_share(df, c, total_rows) <= min_non_null_share:
                    valid = False
                    break
            else:
                # Column expression (expr or derived column)
                try:
                    # Try to add a temporary column name check
                    temp_col_name = f"_tmp_check_{name}"
                    test_df = df.withColumn(temp_col_name, c)
                    if test_df.filter(F.col(temp_col_name).isNotNull()).count() / total_rows <= min_non_null_share:
                        valid = False
                        break
                except Exception:
                    valid = False
                    break
        if valid:
            rules.append((name, block_on(*cols)))

    if "numeric_id" in roles:
         add_rule_if_valid(f"exact_{roles['numeric_id']}", roles['numeric_id'])
         numeric_id_col = roles['numeric_id']
         if numeric_id_col in df.columns and str(df.schema[numeric_id_col].dataType).startswith(("Integer", "Long")):
             mod_col = f"{numeric_id_col}_mod_1000"
             if mod_col not in df.columns:
                 df = df.withColumn(mod_col, col(numeric_id_col) % 1000)
             add_rule_if_valid(f"{numeric_id_col}_mod_1000", mod_col)
         
         numeric_id_hash_col = f"{roles['numeric_id']}_hash"
         if numeric_id_hash_col in df.columns:
             add_rule_if_valid(f"hash_{roles['numeric_id']}", numeric_id_hash_col)

    add_rule_if_valid("exact_email", "email_norm")
    add_rule_if_valid("exact_phone", "phone_digits")
    add_rule_if_valid("phone_last7", "phone_last7")
    add_rule_if_valid("exact_zip", "zip_norm")
    add_rule_if_valid("exact_date", "date_norm")
    add_rule_if_valid("exact_timestamp", "timestamp_norm")
    add_rule_if_valid("timestamp_year_month", expr("substring(timestamp_norm, 1, 7)"))
    add_rule_if_valid("exact_address", "address_norm")
    add_rule_if_valid("address_street", "address_street")
    add_rule_if_valid("exact_geo_lat", "geo_lat_norm")
    add_rule_if_valid("exact_geo_lon", "geo_lon_norm")
    add_rule_if_valid("geo_lat_rounded", "geo_lat_rounded")
    add_rule_if_valid("geo_lon_rounded", "geo_lon_rounded")
    add_rule_if_valid("url_domain_exact", "url_domain")
    add_rule_if_valid("url_subdomain", "url_subdomain")
    
    add_rule_if_valid("exact_first_last_name", "first_name_norm", "last_name_norm")
    add_rule_if_valid("metaphone_full_name", "first_name_metaphone", "last_name_metaphone")
    add_rule_if_valid("exact_last_name_metaphone", "last_name_metaphone")
    add_rule_if_valid("zip_lastname_meta", "zip_norm", "last_name_metaphone")
    add_rule_if_valid("city_firstname", "city_norm", "first_name_norm")
    add_rule_if_valid("first_name_first_char", "first_name_first_char")
    
    if "category_enum" in roles:
        add_rule_if_valid(f"exact_{roles['category_enum']}", roles['category_enum'])
    
    if "text_freeform" in roles:
        text_col = roles['text_freeform']
        if text_col in df.columns:
            hash_col = f"{text_col}_hash"
            if mod_col not in df.columns:
                df = df.withColumn(hash_col, sha2(when(col(text_col).isNull(), lit("")).otherwise(col(text_col).cast("string")), 256))
            add_rule_if_valid(f"hash_{text_col}", hash_col)

    return rules, df

def build_settings_enhanced(df: SparkDataFrame, roles: Dict[str, str], blocking_rules: List[object], total_rows: int) -> SettingsCreator:
    comparisons = []
    def add_comparison_if_valid(comparison_func, col_name: str, *args, **kwargs):
        min_non_null_share = 0.2
        if col_name in df.columns and _nonnull_share(df, col_name, total_rows) > min_non_null_share:
            comparisons.append(comparison_func(col_name, *args, **kwargs))
        else:
            print(f"Skipping comparison for '{col_name}' due to missing column or too many nulls.")

    if "first_name" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "first_name_norm", [0.7, 0.9])
        add_comparison_if_valid(cl.ExactMatch, "first_name_metaphone")
    if "last_name" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "last_name_norm", [0.7, 0.9])
        add_comparison_if_valid(cl.ExactMatch, "last_name_metaphone")
    if "email" in roles:
        add_comparison_if_valid(cl.EmailComparison, "email_norm")
    if "phone" in roles:
        add_comparison_if_valid(cl.LevenshteinAtThresholds, "phone_digits", [2, 4])
    if "address" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "address_norm", [0.7, 0.9])
    if "city" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, "city_norm", [0.8, 0.95])  # Removed term_frequency_adjustments
    if "state" in roles:
        add_comparison_if_valid(cl.ExactMatch, "state_norm")
    if "zip" in roles:
        add_comparison_if_valid(cl.PostcodeComparison, "zip_norm")
    if "date" in roles:
        add_comparison_if_valid(cl.DateOfBirthComparison, "date_norm", input_is_string=True)
    if "full_name" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, roles["full_name"], [0.8, 0.95])
    if "text_freeform" in roles:
        add_comparison_if_valid(cl.JaroWinklerAtThresholds, roles["text_freeform"], [0.8, 0.95])
    if "url" in roles:
        add_comparison_if_valid(cl.ExactMatch, "url_domain")
    if not comparisons:
        if "numeric_id" in roles:
            numeric_id_hash_col = f"{roles['numeric_id']}_hash"
            if numeric_id_hash_col in df.columns and _nonnull_share(df, numeric_id_hash_col, total_rows) > 0.2:
                add_comparison_if_valid(cl.ExactMatch, numeric_id_hash_col)
        if not comparisons:
            raise ValueError(
                "Splink model cannot be trained because no valid comparison columns were generated. "
                f"The table may lack suitable fields for deduplication. Detected roles: {list(roles.keys())}"
            )

    return SettingsCreator(
        link_type="dedupe_only",
        em_convergence=0.001,
        max_iterations=25,
        comparisons=comparisons,
        blocking_rules_to_generate_predictions=blocking_rules,
        retain_intermediate_calculation_columns=False,
    )

def select_optimal_blocking_rules(
    df: SparkDataFrame,
    candidate_rules: List[Tuple[str, object]],
    db_api: SparkAPI,
    max_rules: Optional[int] = 5,
    max_comparisons: int = 20_000_000,
    min_comparisons: int = 100,
    ideal_comparisons: int = 1_000_000,
    verbosity: bool = True
) -> Tuple[List[Tuple[str, object]], List[Dict]]:
    """
    Selects an optimal set of blocking rules by evaluating the number of comparisons in parallel.
    Selects rules from the middle layer of comparison counts, excluding those with too few or too many pairs.
    """
    diagnostics = []
    scored = []

    def compute_count(name, rule):
        try:
            result = count_comparisons_from_blocking_rule(
                table_or_tables=df, blocking_rule=rule, link_type="dedupe_only", db_api=db_api
            )
            cnt = int(result.get("number_of_comparisons_to_be_scored_post_filter_conditions", float('inf')))
            return name, rule, cnt
        except Exception as e:
            if verbosity:
                print(f"Warning: Failed to evaluate blocking rule '{name}': {e}")
            return name, rule, -1

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(compute_count, name, rule) for name, rule in candidate_rules]
        for future in concurrent.futures.as_completed(futures):
            name, rule, cnt = future.result()
            reason = "not_evaluated"
            if cnt == float('inf'):
                reason = "not_evaluated"
            elif cnt == 0:
                reason = "zero_comparisons"
            elif cnt < min_comparisons:
                reason = "too_few_comparisons"
            elif cnt > max_comparisons:
                reason = "too_many_comparisons"
            else:
                reason = "selected"
            diagnostics.append({"name": name, "comparisons": (f"{cnt:,}" if isinstance(cnt, int) and cnt >= 0 else str(cnt)), "kept": reason == "selected", "reason": reason, "rule": rule})
            if reason == "selected":
                scored.append((name, rule, cnt))

    scored.sort(key=lambda x: x[2])
    if len(scored) > max_rules:
        start = (len(scored) - max_rules) // 2
        scored = scored[start:start + max_rules]
    selected_rules_with_names = [(name, rule) for name, rule, cnt in scored]

    if not selected_rules_with_names and candidate_rules:
        fallback_rule_info = None
        min_diff = float('inf')
        for diag in diagnostics:
            if isinstance(diag['comparisons'], int) and diag['comparisons'] > 0:
                diff = abs(diag['comparisons'] - ideal_comparisons)
                if diff < min_diff:
                    fallback_rule_info = (diag['name'], diag['rule'], diag['comparisons'])
                    min_diff = diff
        if fallback_rule_info:
            selected_rules_with_names.append((fallback_rule_info[0], fallback_rule_info[1]))
            for diag in diagnostics:
                if diag['name'] == fallback_rule_info[0]:
                    diag['kept'] = True
                    diag['reason'] = "fallback_closest_to_ideal"
                    break
        else:
            selected_rules_with_names.append(candidate_rules[0])
            for diag in diagnostics:
                if diag['name'] == candidate_rules[0][0]:
                    diag['kept'] = True
                    diag['reason'] = "fallback_first_candidate"
                    break
            else:
                diagnostics.append({"name": candidate_rules[0][0], "comparisons": "unknown", "kept": True, "reason": "fallback_first_candidate", "rule": candidate_rules[0][1]})

    return selected_rules_with_names, diagnostics

def auto_generate_settings(
    df: SparkDataFrame, 
    db_api: SparkAPI, 
    max_rules: Optional[int] = 5, 
    universal_mode: bool = True, 
    spark=None,
    sample_size: int = 10000,
    rows_per_partition: int = 100000,
    max_comparisons: int = 20_000_000,
    max_comparisons_cap: int = 1_000_000_000,
    verbosity: bool = True
) -> Tuple[Dict, Dict, List, SparkDataFrame]:
    """
    Main orchestration function to automatically generate Splink settings from a DataFrame.
    Compatible with Splink using SparkAPI, optimized for laptop execution.
    """
    try:
        if df.isEmpty() or df.count() < 2:
            raise ValueError("Input DataFrame must have at least two rows to perform deduplication.")

        if spark is None:
            raise ValueError("SparkSession 'spark' must be provided as an argument or defined in the global scope.")

        total_rows = df.count()
        num_partitions = max(16, int(total_rows / rows_per_partition) + 1)
        df = df.repartition(num_partitions)
        spark.conf.set("spark.sql.shuffle.partitions", num_partitions)

        spark.sparkContext._conf.set("spark.python.worker.memory", "2g")
        spark.sparkContext._conf.set("spark.driver.maxResultSize", "4g")
        spark.sparkContext._conf.set("spark.executor.memory", "8g")
        spark.sparkContext._conf.set("spark.executor.cores", "2")
        spark.sparkContext._conf.set("spark.executor.instances", "4")

        if verbosity:
            print(f"Dataset size: {total_rows} rows. Using {num_partitions} partitions.")

        roles = infer_roles_enhanced(df, sample_size=sample_size, verbosity=verbosity)
        if verbosity:
            print(f"✅ Detected roles: {roles}")
        
        df_enhanced = ensure_derived_columns_enhanced(df, roles, total_rows=total_rows, verbosity=verbosity)
        
        candidate_rules, df_enhanced = generate_robust_blocking_rules(df_enhanced, roles, total_rows=total_rows)
        if verbosity:
            print(f"\n⚙️  Generated {len(candidate_rules)} candidate blocking rules.")
        
        # Dynamic comparison thresholds
        scaling_factor = max(1, total_rows // 1_000_000)
        max_comparisons = min(max_comparisons_cap, 20_000_000 * scaling_factor)
        min_comparisons = max(100, int(total_rows * 0.0001))
        ideal_comparisons = max(1_000_000, int(total_rows * 0.001))
        if verbosity:
            print(f"Dynamic thresholds: max_comparisons={max_comparisons:,}, min_comparisons={min_comparisons:,}, ideal_comparisons={ideal_comparisons:,} (scaling_factor: {scaling_factor})")
        if max_comparisons > 1_000_000_000 and verbosity:
            print("Warning: High comparison count may require larger Spark cluster.")
        
        selected_rules, diagnostics = select_optimal_blocking_rules(
            df_enhanced, candidate_rules, db_api, 
            max_rules=max_rules, 
            max_comparisons=max_comparisons, 
            min_comparisons=min_comparisons,
            ideal_comparisons=ideal_comparisons,
            verbosity=verbosity
        )
        
        if verbosity:
            print("\n📊 Blocking rule analysis:")
            for diag in diagnostics:
                status = "✓" if diag["kept"] else "✗"
                reason_detail = f" ({diag['reason']})" if not diag["kept"] else ""
                comps = f"{diag['comparisons']:,}" if isinstance(diag['comparisons'], int) and diag['comparisons'] >= 0 else diag['comparisons']
                print(f"{status} {diag['name']}: {comps} comparisons{reason_detail}")

        final_blocking_rules = [rule for _, rule in selected_rules]
        settings_creator = build_settings_enhanced(df_enhanced, roles, final_blocking_rules, total_rows=total_rows)
        if verbosity:
            print("\n✅ Successfully generated complete Splink settings.")
        
        settings_dict = None
        if isinstance(settings_creator, dict):
            settings_dict = settings_creator
        elif hasattr(settings_creator, "as_dict") and callable(settings_creator.as_dict):
            settings_dict = settings_creator.as_dict()
        elif hasattr(settings_creator, "settings_dict"):
            settings_dict = settings_creator.settings_dict
        elif hasattr(settings_creator, "to_dict") and callable(settings_creator.to_dict):
            settings_dict = settings_creator.to_dict()
        elif hasattr(settings_creator, "create_settings_dict") and callable(settings_creator.create_settings_dict):
            settings_dict = settings_creator.create_settings_dict(sql_dialect_str="spark")
        else:
            raise TypeError(
                f"Could not convert Splink SettingsCreator object ({type(settings_creator)}) "
                f"to a dictionary. This is likely due to an incompatible Splink library version. "
                f"Available attributes: {dir(settings_creator)}"
            )

        return settings_dict, roles, diagnostics, df_enhanced
    
    except Exception as e:
        if verbosity:
            print(f"Error in auto_generate_settings: {e}")
        raise

class BlockingBotSpark:
    def __init__(self, db_api: SparkAPI):
        """
        Initializes the BlockingBotSpark with a SparkAPI object.
        """
        self.db_api = db_api

    def auto_generate_settings(self, df: SparkDataFrame, max_rules: Optional[int] = 5, max_comparisons: int = 20_000_000, max_comparisons_cap: int = 1_000_000_000):
        return auto_generate_settings(df, self.db_api, max_rules=max_rules, max_comparisons=max_comparisons, max_comparisons_cap=max_comparisons_cap)