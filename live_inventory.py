from pathlib import Path
from datetime import date
import uuid

import pandas as pd


# File paths

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

PRODUCTS_FILE = DATA_DIR / "live_products.csv"
INVENTORY_FILE = DATA_DIR / "live_inventory.csv"
SALES_FILE = DATA_DIR / "live_sales.csv"

HISTORICAL_FILE = DATA_DIR / "processed" / "cleaned_data.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# CSV columns

PRODUCT_COLUMNS = [
    "product_id",
    "product_name",
    "category",
    "storage_type",
    "shelf_life_days",
    "cost_price",
    "selling_price",
    "created_at",
]

INVENTORY_COLUMNS = [
    "batch_id",
    "product_id",
    "quantity",
    "received_date",
    "expiration_date",
    "cost_price",
    "selling_price",
    "storage_type",
    "created_at",
]

SALES_COLUMNS = [
    "sale_id",
    "product_id",
    "quantity_sold",
    "sale_date",
    "selling_price",
    "created_at",
]


def _now():
    return pd.Timestamp.now().isoformat()


def _load_csv(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = None

    return df


def load_live_products():
    return _load_csv(PRODUCTS_FILE, PRODUCT_COLUMNS)


def load_live_inventory():
    df = _load_csv(INVENTORY_FILE, INVENTORY_COLUMNS)

    if not df.empty:
        df["quantity"] = pd.to_numeric(
            df["quantity"],
            errors="coerce"
        ).fillna(0)

        df["expiration_date"] = pd.to_datetime(
            df["expiration_date"],
            errors="coerce"
        )

        df["received_date"] = pd.to_datetime(
            df["received_date"],
            errors="coerce"
        )

    return df


def load_live_sales():
    return _load_csv(SALES_FILE, SALES_COLUMNS)


# Historical product information

def load_historical_products():
    if not HISTORICAL_FILE.exists():
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "category",
            ]
        )

    df = pd.read_csv(HISTORICAL_FILE)

    if "product_id" not in df.columns:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "category",
            ]
        )

    df["product_id"] = df["product_id"].astype(str).str.strip()

    useful_columns = ["product_id"]

    for column in [
        "product_name",
        "category",
        "product_category",
        "storage_type",
        "shelf_life_days",
        "cost_price",
        "selling_price",
    ]:
        if column in df.columns:
            useful_columns.append(column)

    result = (
        df[useful_columns]
        .drop_duplicates(
            subset=["product_id"],
            keep="last"
        )
        .copy()
    )

    if (
        "category" not in result.columns
        and "product_category" in result.columns
    ):
        result["category"] = result["product_category"]

    return result


# Combine historical and live products

def get_all_products():
    historical = load_historical_products()
    live = load_live_products()

    frames = []

    if not historical.empty:
        frames.append(historical)

    if not live.empty:
        frames.append(live)

    if not frames:
        return pd.DataFrame(
            columns=[
                "product_id",
                "product_name",
                "category",
            ]
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    combined["product_id"] = (
        combined["product_id"]
        .astype(str)
        .str.strip()
    )

    combined = (
        combined
        .drop_duplicates(
            subset=["product_id"],
            keep="last"
        )
        .sort_values("product_id")
        .reset_index(drop=True)
    )

    return combined


def product_exists(product_id):
    product_id = str(product_id).strip()
    products = get_all_products()

    if products.empty:
        return False

    return (
        products["product_id"]
        .astype(str)
        .str.strip()
        .eq(product_id)
        .any()
    )


def get_product(product_id):
    product_id = str(product_id).strip()
    products = get_all_products()

    match = products[
        products["product_id"]
        .astype(str)
        .str.strip()
        == product_id
    ]

    if match.empty:
        return None

    return match.iloc[-1].to_dict()


# Add a brand-new product

def add_new_product(
    product_id,
    product_name,
    category,
    storage_type,
    shelf_life_days,
    cost_price,
    selling_price,
    initial_quantity,
    received_date,
    expiration_date,
):
    product_id = str(product_id).strip()

    if not product_id:
        raise ValueError("Product ID is required.")

    if product_exists(product_id):
        raise ValueError(
            "This Product ID already exists. "
            "Use the Restock Existing Product option."
        )

    if initial_quantity <= 0:
        raise ValueError(
            "Initial quantity must be greater than zero."
        )

    if pd.Timestamp(expiration_date) < pd.Timestamp(received_date):
        raise ValueError(
            "Expiration date cannot be before received date."
        )

    products = load_live_products()

    new_product = pd.DataFrame(
        [
            {
                "product_id": product_id,
                "product_name": product_name.strip(),
                "category": category.strip(),
                "storage_type": storage_type.strip(),
                "shelf_life_days": int(shelf_life_days),
                "cost_price": float(cost_price),
                "selling_price": float(selling_price),
                "created_at": _now(),
            }
        ]
    )

    products = pd.concat(
        [products, new_product],
        ignore_index=True
    )

    products.to_csv(PRODUCTS_FILE, index=False)

    # Create the first inventory batch for the product
    add_inventory_batch(
        product_id=product_id,
        quantity=initial_quantity,
        received_date=received_date,
        expiration_date=expiration_date,
        cost_price=cost_price,
        selling_price=selling_price,
        storage_type=storage_type,
    )

    return product_id


# Restock an existing product

def add_inventory_batch(
    product_id,
    quantity,
    received_date,
    expiration_date,
    cost_price,
    selling_price,
    storage_type,
):
    product_id = str(product_id).strip()

    if not product_exists(product_id):
        raise ValueError(
            "Product does not exist. "
            "Add it as a brand-new product first."
        )

    if quantity <= 0:
        raise ValueError(
            "Restock quantity must be greater than zero."
        )

    if pd.Timestamp(expiration_date) < pd.Timestamp(received_date):
        raise ValueError(
            "Expiration date cannot be before received date."
        )

    inventory = load_live_inventory()

    batch_id = (
        "BATCH_"
        + uuid.uuid4().hex[:10].upper()
    )

    new_batch = pd.DataFrame(
        [
            {
                "batch_id": batch_id,
                "product_id": product_id,
                "quantity": float(quantity),
                "received_date": str(received_date),
                "expiration_date": str(expiration_date),
                "cost_price": float(cost_price),
                "selling_price": float(selling_price),
                "storage_type": storage_type.strip(),
                "created_at": _now(),
            }
        ]
    )

    inventory = pd.concat(
        [inventory, new_batch],
        ignore_index=True
    )

    inventory.to_csv(INVENTORY_FILE, index=False)

    return batch_id


# Current stock

def get_current_stock(product_id=None):
    inventory = load_live_inventory()

    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "current_quantity",
            ]
        )

    inventory = inventory[
        inventory["quantity"] > 0
    ].copy()

    if product_id is not None:
        product_id = str(product_id).strip()

        inventory = inventory[
            inventory["product_id"]
            .astype(str)
            .str.strip()
            == product_id
        ]

    if inventory.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "current_quantity",
            ]
        )

    grouped = (
        inventory.groupby(
            "product_id",
            as_index=False
        )["quantity"]
        .sum()
        .rename(
            columns={
                "quantity": "current_quantity"
            }
        )
    )

    return grouped


def get_product_batches(product_id):
    inventory = load_live_inventory()

    if inventory.empty:
        return inventory

    result = inventory[
        inventory["product_id"]
        .astype(str)
        .str.strip()
        == str(product_id).strip()
    ].copy()

    result = result[
        result["quantity"] > 0
    ]

    result = result.sort_values(
        "expiration_date"
    )

    return result


# Record a sale using FEFO

def record_sale(
    product_id,
    quantity_sold,
    sale_date=None,
    selling_price=None,
):
    product_id = str(product_id).strip()

    if quantity_sold <= 0:
        raise ValueError(
            "Sale quantity must be greater than zero."
        )

    inventory = load_live_inventory()

    if inventory.empty:
        raise ValueError(
            "There is no live inventory available."
        )

    mask = (
        inventory["product_id"]
        .astype(str)
        .str.strip()
        == product_id
    )

    product_batches = inventory[
        mask
    ].copy()

    product_batches["quantity"] = pd.to_numeric(
        product_batches["quantity"],
        errors="coerce"
    ).fillna(0)

    available_quantity = (
        product_batches["quantity"]
        .clip(lower=0)
        .sum()
    )

    if quantity_sold > available_quantity:
        raise ValueError(
            f"Not enough stock. "
            f"Available quantity is "
            f"{available_quantity:.2f}."
        )

    # FEFO: sell from the batch with the earliest expiration date first
    product_batches["expiration_date"] = pd.to_datetime(
        product_batches["expiration_date"],
        errors="coerce"
    )

    product_batches = product_batches.sort_values(
        "expiration_date"
    )

    remaining_to_sell = float(quantity_sold)

    for index in product_batches.index:
        if remaining_to_sell <= 0:
            break

        batch_quantity = float(
            inventory.at[index, "quantity"]
        )

        used = min(
            batch_quantity,
            remaining_to_sell
        )

        inventory.at[index, "quantity"] = (
            batch_quantity - used
        )

        remaining_to_sell -= used

    inventory.to_csv(
        INVENTORY_FILE,
        index=False
    )

    # Save the sale in the sales log
    sales = load_live_sales()

    if sale_date is None:
        sale_date = date.today()

    if selling_price is None:
        product = get_product(product_id)

        if product:
            selling_price = product.get(
                "selling_price",
                0
            )
        else:
            selling_price = 0

    sale_id = (
        "SALE_"
        + uuid.uuid4().hex[:10].upper()
    )

    sale_record = pd.DataFrame(
        [
            {
                "sale_id": sale_id,
                "product_id": product_id,
                "quantity_sold": float(quantity_sold),
                "sale_date": str(sale_date),
                "selling_price": float(selling_price),
                "created_at": _now(),
            }
        ]
    )

    sales = pd.concat(
        [sales, sale_record],
        ignore_index=True
    )

    sales.to_csv(
        SALES_FILE,
        index=False
    )

    return sale_id


# Snapshot used by the AI system

def get_live_inventory_snapshot():
    products = get_all_products()
    inventory = load_live_inventory()

    if inventory.empty:
        return pd.DataFrame()

    inventory = inventory[
        inventory["quantity"] > 0
    ].copy()

    if inventory.empty:
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()

    inventory["days_to_expiry"] = (
        inventory["expiration_date"]
        - today
    ).dt.days

    # Total active quantity for each product
    totals = (
        inventory.groupby(
            "product_id",
            as_index=False
        )["quantity"]
        .sum()
        .rename(
            columns={
                "quantity": "current_quantity"
            }
        )
    )

    # Earliest active batch for each product
    earliest = (
        inventory
        .sort_values("expiration_date")
        .groupby(
            "product_id",
            as_index=False
        )
        .first()
    )

    snapshot = totals.merge(
        earliest[
            [
                "product_id",
                "expiration_date",
                "days_to_expiry",
                "selling_price",
                "cost_price",
                "storage_type",
            ]
        ],
        on="product_id",
        how="left",
    )

    if not products.empty:
        useful_product_columns = [
            column
            for column in [
                "product_id",
                "product_name",
                "category",
                "shelf_life_days",
            ]
            if column in products.columns
        ]

        snapshot = snapshot.merge(
            products[useful_product_columns],
            on="product_id",
            how="left",
        )

    return snapshot