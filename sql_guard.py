import re

# pattern to allow only SELECT statements
_ALLOWED_PREFIX = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
# pattern to disallow destructive SQL keywords
_FORBIDDEN = re.compile(r"\b(update|insert|delete|drop|alter|truncate|attach|pragma)\b", re.IGNORECASE)


def is_safe_select(sql: str) -> bool:
    """Return True if the SQL is a safe SELECT query (no forbidden keywords)."""
    return bool(_ALLOWED_PREFIX.search(sql)) and not bool(_FORBIDDEN.search(sql))


def sanitize_limit(sql: str, default_limit: int = 50) -> str:
    """Append a LIMIT clause if none exists to prevent unbounded results."""
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {default_limit};"
