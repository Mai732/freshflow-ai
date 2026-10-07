from pathlib import Path
import os
import subprocess
import sys
import time

import pandas as pd
import requests
import streamlit as st


# PROJECT IMPORTS

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:

    sys.path.append(
        str(BASE_DIR)
    )


from live_inventory import (
    add_new_product,
    add_inventory_batch,
    record_sale,
    get_all_products,
    get_current_stock,
    get_product_batches,
    get_live_inventory_snapshot,
    get_product
)


# PAGE CONFIG

st.set_page_config(
    page_title="FreshFlow AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CONFIG

API_URL = os.getenv(
    "FRESHFLOW_API_URL",
    "http://127.0.0.1:8000"
)


# CSS

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    section[data-testid="stSidebar"] {
        border-right:
            1px solid rgba(128,128,128,0.15);
    }

    div[data-testid="stMetric"] {
        border:
            1px solid rgba(128,128,128,0.18);
        border-radius: 15px;
        padding: 12px;
    }

    .action-box {
        padding: 16px 20px;
        border-radius: 12px;
        border-left: 5px solid #2f8f65;
        background: rgba(47,143,101,0.08);
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# FASTAPI

def api_is_running():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=2
        )

        return (
            response.status_code
            == 200
        )

    except Exception:

        return False


@st.cache_resource
def start_fastapi_backend():

    if api_is_running():

        return None

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000"
    ]

    kwargs = {
        "cwd":
            str(BASE_DIR),

        "stdout":
            subprocess.DEVNULL,

        "stderr":
            subprocess.DEVNULL
    }

    if os.name == "nt":

        kwargs[
            "creationflags"
        ] = getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0
        )

    process = subprocess.Popen(
        command,
        **kwargs
    )

    for _ in range(30):

        time.sleep(
            0.5
        )

        if api_is_running():

            break

    return process


#start_fastapi_backend()


# SESSION STATE

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role":
                "assistant",

            "content":
                (
                    "Hello — I'm **FreshFlow AI**. 🌿\n\n"
                    "Ask me about product demand, "
                    "expiration, waste risk, discounts, "
                    "order adjustments, inventory, or "
                    "FDA food-storage guidance."
                ),

            "payload":
                None
        }
    ]


# CHAT API

def call_chat_api(
    message
):

    history = []

    for item in (
        st.session_state.messages[
            -8:
        ]
    ):

        history.append(
            {
                "role":
                    item[
                        "role"
                    ],

                "content":
                    item[
                        "content"
                    ]
            }
        )

    response = requests.post(
        f"{API_URL}/chat",

        json={
            "message":
                message,

            "history":
                history
        },

        timeout=180
    )

    response.raise_for_status()

    return response.json()


# PRODUCT ANALYSIS OUTPUT

def render_product_analysis(
    data
):

    if not data:
        return

    st.markdown(
        f"#### 📦 {data.get('product_id', 'Product')}"
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Current Quantity",
        f"{float(data.get('current_quantity', 0)):,.1f}"
    )

    c2.metric(
        "Predicted Demand",
        f"{float(data.get('predicted_demand', 0)):,.1f}"
    )

    c3.metric(
        "Expected Surplus",
        f"{float(data.get('expected_surplus', 0)):,.1f}"
    )

    c4.metric(
        "Days to Expiry",
        data.get(
            "days_to_expiry",
            "-"
        )
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "Expiration Risk",
        data.get(
            "expiration_risk",
            "-"
        )
    )

    c2.metric(
        "Waste Risk",
        data.get(
            "waste_risk",
            "-"
        )
    )

    c3.metric(
        "Discount",
        f"{int(data.get('recommended_discount', 0))}%"
    )

    adjustment = int(
        round(
            float(
                data.get(
                    "recommended_order_adjustment",
                    0
                )
            )
        )
    )

    c4.metric(
        "Order Adjustment",
        f"{adjustment:+d} units"
    )

    action = (
        data.get(
            "recommended_action",
            "NO_ACTION"
        )
        .replace(
            "_",
            " "
        )
        .title()
    )

    st.markdown(
        f"""
        <div class="action-box">
            <b>Recommended Action</b><br>
            {action}
        </div>
        """,
        unsafe_allow_html=True
    )


# FORECAST

def render_forecast(
    data
):

    if not data:
        return

    forecast = data.get(
        "daily_forecast",
        []
    )

    if not forecast:
        return

    forecast_df = pd.DataFrame(
        forecast
    )

    st.metric(
        "Total Predicted Demand",
        f"{float(data.get('total_predicted_demand', 0)):,.2f} units"
    )

    if (
        "date"
        in forecast_df.columns
        and
        "predicted_demand"
        in forecast_df.columns
    ):

        chart_df = (
            forecast_df
            .copy()
        )

        chart_df[
            "date"
        ] = pd.to_datetime(
            chart_df[
                "date"
            ]
        )

        chart_df = (
            chart_df
            .set_index(
                "date"
            )
        )

        st.line_chart(
            chart_df[
                [
                    "predicted_demand"
                ]
            ]
        )

    st.dataframe(
        forecast_df,
        use_container_width=True,
        hide_index=True
    )


# PRODUCT LIST

def render_product_list(
    data
):

    if not data:
        return

    result_df = pd.DataFrame(
        data
    )

    if result_df.empty:

        st.info(
            "No matching products found."
        )

        return

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )


# SOURCES

def render_sources(
    sources
):

    if not sources:
        return

    with st.expander(
        "📚 FDA evidence used"
    ):

        for index, source in enumerate(
            sources,
            start=1
        ):

            st.markdown(
                f"**Source {index}**"
            )

            if isinstance(
                source,
                dict
            ):

                st.caption(
                    (
                        f"{source.get('source', 'FDA')} "
                        f"— page "
                        f"{source.get('page', '?')}"
                    )
                )

                st.write(
                    source.get(
                        "text",
                        ""
                    )
                )

            else:

                st.write(
                    source
                )

            st.divider()


# PAYLOAD

def render_payload(
    payload
):

    if not payload:
        return

    intent = (
        payload.get(
            "intent"
        )
    )

    data = payload.get(
        "data"
    )

    if intent == "ANALYZE_PRODUCT":

        render_product_analysis(
            data
        )

    elif intent == "PREDICT_DEMAND":

        render_forecast(
            data
        )

    elif intent in [
        "EXPIRING_PRODUCTS",
        "HIGH_RISK_PRODUCTS",
        "DISCOUNT_CANDIDATES"
    ]:

        render_product_list(
            data
        )

    render_sources(
        payload.get(
            "sources",
            []
        )
    )


# SIDEBAR

with st.sidebar:

    st.title(
        "🌿 FreshFlow AI"
    )

    st.caption(
        "Supermarket Food Waste Assistant"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "💬 AI Assistant",
            "📦 Inventory Management"
        ]
    )

    st.divider()

    if page == "💬 AI Assistant":

        st.markdown(
            "### Try asking"
        )

        suggestions = [
            "Which products should I discount?",
            "Which products expire within 3 days?",
            "Which products have the highest waste risk?",
            "Analyze BAK_BAG_101",
            "Predict demand for BAK_BAG_101 for 5 days",
            "What does the FDA say about date marking?"
        ]

        for index, suggestion in enumerate(
            suggestions
        ):

            if st.button(
                suggestion,
                key=
                    f"suggest_{index}",
                use_container_width=True
            ):

                st.session_state[
                    "pending_prompt"
                ] = suggestion

                st.rerun()

        st.divider()

        if st.button(
            "＋ New Chat",
            use_container_width=True
        ):

            st.session_state.messages = [
                {
                    "role":
                        "assistant",

                    "content":
                        (
                            "New conversation started. "
                            "What would you like to check?"
                        ),

                    "payload":
                        None
                }
            ]

            st.rerun()


# PAGE TITLE - NO HTML

st.title(
    "🌿 FreshFlow AI"
)

if page == "💬 AI Assistant":

    st.caption(
        "Ask about inventory, demand, expiration, "
        "waste risk, discounts, ordering, and FDA guidance."
    )

else:

    st.caption(
        "Add products, restock existing products, "
        "record sales, and monitor current inventory."
    )


# AI ASSISTANT PAGE

if page == "💬 AI Assistant":

    for message in (
        st.session_state.messages
    ):

        with st.chat_message(
            message[
                "role"
            ]
        ):

            st.markdown(
                message[
                    "content"
                ]
            )

            if (
                message[
                    "role"
                ]
                == "assistant"
            ):

                render_payload(
                    message.get(
                        "payload"
                    )
                )

    prompt = st.chat_input(
        "Ask FreshFlow about your supermarket..."
    )

    if (
        "pending_prompt"
        in st.session_state
    ):

        prompt = (
            st.session_state
            .pop(
                "pending_prompt"
            )
        )

    if prompt:

        st.session_state.messages.append(
            {
                "role":
                    "user",

                "content":
                    prompt,

                "payload":
                    None
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                prompt
            )

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "FreshFlow is analyzing..."
            ):

                try:

                    result = (
                        call_chat_api(
                            prompt
                        )
                    )

                    answer = (
                        result.get(
                            "answer",
                            "Analysis completed."
                        )
                    )

                    st.markdown(
                        answer
                    )

                    render_payload(
                        result
                    )

                    st.session_state.messages.append(
                        {
                            "role":
                                "assistant",

                            "content":
                                answer,

                            "payload":
                                result
                        }
                    )

                except Exception as error:

                    st.error(
                        "The analysis could not be completed."
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            str(
                                error
                            )
                        )


# INVENTORY MANAGEMENT PAGE

else:

    inventory_tab1, inventory_tab2, inventory_tab3, inventory_tab4 = (
        st.tabs(
            [
                "➕ Add New Product",
                "📥 Restock Existing Product",
                "🛒 Record Sale",
                "📊 Current Inventory"
            ]
        )
    )



    # ADD BRAND-NEW PRODUCT


    with inventory_tab1:

        st.subheader(
            "Add a Brand-New Product"
        )

        st.info(
            "Use this when the supermarket has never "
            "carried this product before."
        )

        with st.form(
            "new_product_form",
            clear_on_submit=True
        ):

            col1, col2 = (
                st.columns(2)
            )

            with col1:

                new_product_id = (
                    st.text_input(
                        "Product ID *",
                        placeholder=
                            "Example: DAI_MILK_001"
                    )
                )

                new_product_name = (
                    st.text_input(
                        "Product Name *",
                        placeholder=
                            "Example: Fresh Whole Milk"
                    )
                )

                new_category = (
                    st.text_input(
                        "Category *",
                        placeholder=
                            "Example: Dairy"
                    )
                )

                new_storage = (
                    st.selectbox(
                        "Storage Type",
                        [
                            "Refrigerated",
                            "Frozen",
                            "Room Temperature",
                            "Other"
                        ]
                    )
                )

                new_shelf_life = (
                    st.number_input(
                        "Shelf Life (days)",
                        min_value=1,
                        value=7,
                        step=1
                    )
                )

            with col2:

                new_quantity = (
                    st.number_input(
                        "Initial Quantity *",
                        min_value=1.0,
                        value=10.0,
                        step=1.0
                    )
                )

                new_cost = (
                    st.number_input(
                        "Cost Price",
                        min_value=0.0,
                        value=0.0,
                        step=0.5
                    )
                )

                new_price = (
                    st.number_input(
                        "Selling Price",
                        min_value=0.0,
                        value=0.0,
                        step=0.5
                    )
                )

                new_received = (
                    st.date_input(
                        "Received Date"
                    )
                )

                new_expiration = (
                    st.date_input(
                        "Expiration Date"
                    )
                )

            new_submit = (
                st.form_submit_button(
                    "Add Product and First Batch",
                    use_container_width=True,
                    type="primary"
                )
            )

        if new_submit:

            try:

                add_new_product(
                    product_id=
                        new_product_id,

                    product_name=
                        new_product_name,

                    category=
                        new_category,

                    storage_type=
                        new_storage,

                    shelf_life_days=
                        new_shelf_life,

                    cost_price=
                        new_cost,

                    selling_price=
                        new_price,

                    initial_quantity=
                        new_quantity,

                    received_date=
                        new_received,

                    expiration_date=
                        new_expiration
                )

                st.success(
                    (
                        f"✅ {new_product_name} "
                        f"({new_product_id}) was added successfully."
                    )
                )

                st.cache_data.clear()

            except Exception as error:

                st.error(
                    str(
                        error
                    )
                )


  
    # RESTOCK EXISTING PRODUCT
  

    with inventory_tab2:

        st.subheader(
            "Restock Existing Product"
        )

        st.info(
            "Every restock is stored as a separate batch "
            "because the new stock can have a different "
            "expiration date."
        )

        products = (
            get_all_products()
        )

        if products.empty:

            st.warning(
                "No products are available."
            )

        else:

            product_options = (
                products[
                    "product_id"
                ]
                .astype(str)
                .tolist()
            )

            restock_product = (
                st.selectbox(
                    "Select Existing Product",
                    product_options,
                    key=
                        "restock_product"
                )
            )

            product_info = (
                get_product(
                    restock_product
                )
            )

            if product_info:

                product_name = (
                    product_info.get(
                        "product_name"
                    )
                )

                if (
                    product_name
                    and
                    str(
                        product_name
                    )
                    != "nan"
                ):

                    st.caption(
                        f"Product: {product_name}"
                    )

            with st.form(
                "restock_form",
                clear_on_submit=True
            ):

                col1, col2 = (
                    st.columns(2)
                )

                with col1:

                    restock_quantity = (
                        st.number_input(
                            "New Quantity *",
                            min_value=1.0,
                            value=10.0,
                            step=1.0
                        )
                    )

                    restock_received = (
                        st.date_input(
                            "Received Date",
                            key=
                                "restock_received"
                        )
                    )

                    restock_expiration = (
                        st.date_input(
                            "Expiration Date",
                            key=
                                "restock_expiration"
                        )
                    )

                with col2:

                    default_cost = 0.0
                    default_price = 0.0
                    default_storage = (
                        "Refrigerated"
                    )

                    if product_info:

                        try:

                            default_cost = float(
                                product_info.get(
                                    "cost_price",
                                    0
                                )
                            )

                            if pd.isna(
                                default_cost
                            ):

                                default_cost = 0.0

                        except Exception:

                            default_cost = 0.0

                        try:

                            default_price = float(
                                product_info.get(
                                    "selling_price",
                                    0
                                )
                            )

                            if pd.isna(
                                default_price
                            ):

                                default_price = 0.0

                        except Exception:

                            default_price = 0.0

                        storage_value = (
                            product_info.get(
                                "storage_type"
                            )
                        )

                        if (
                            storage_value
                            and
                            str(
                                storage_value
                            )
                            != "nan"
                        ):

                            default_storage = str(
                                storage_value
                            )

                    restock_cost = (
                        st.number_input(
                            "Cost Price",
                            min_value=0.0,
                            value=
                                default_cost,
                            step=0.5,
                            key=
                                "restock_cost"
                        )
                    )

                    restock_price = (
                        st.number_input(
                            "Selling Price",
                            min_value=0.0,
                            value=
                                default_price,
                            step=0.5,
                            key=
                                "restock_price"
                        )
                    )

                    restock_storage = (
                        st.text_input(
                            "Storage Type",
                            value=
                                default_storage
                        )
                    )

                restock_submit = (
                    st.form_submit_button(
                        "Add Restock Batch",
                        use_container_width=True,
                        type="primary"
                    )
                )

            if restock_submit:

                try:

                    batch_id = (
                        add_inventory_batch(
                            product_id=
                                restock_product,

                            quantity=
                                restock_quantity,

                            received_date=
                                restock_received,

                            expiration_date=
                                restock_expiration,

                            cost_price=
                                restock_cost,

                            selling_price=
                                restock_price,

                            storage_type=
                                restock_storage
                        )
                    )

                    st.success(
                        (
                            f"✅ Restock saved successfully. "
                            f"Batch: {batch_id}"
                        )
                    )

                    st.cache_data.clear()

                except Exception as error:

                    st.error(
                        str(
                            error
                        )
                    )


  
    # RECORD SALE
  

    with inventory_tab3:

        st.subheader(
            "Record Product Sale"
        )

        st.caption(
            "FreshFlow automatically removes sold stock "
            "from the earliest-expiring batch first (FEFO)."
        )

        current_stock = (
            get_current_stock()
        )

        if current_stock.empty:

            st.info(
                "There is no live inventory to sell yet."
            )

        else:

            available_products = (
                current_stock[
                    current_stock[
                        "current_quantity"
                    ] > 0
                ][
                    "product_id"
                ]
                .astype(str)
                .tolist()
            )

            sale_product = (
                st.selectbox(
                    "Product",
                    available_products,
                    key=
                        "sale_product"
                )
            )

            available_row = (
                current_stock[
                    current_stock[
                        "product_id"
                    ]
                    .astype(str)
                    ==
                    sale_product
                ]
            )

            available_amount = float(
                available_row[
                    "current_quantity"
                ]
                .iloc[0]
            )

            st.metric(
                "Available Stock",
                f"{available_amount:,.1f} units"
            )

            with st.form(
                "sale_form",
                clear_on_submit=True
            ):

                sale_quantity = (
                    st.number_input(
                        "Quantity Sold",
                        min_value=1.0,
                        max_value=
                            max(
                                1.0,
                                available_amount
                            ),
                        value=1.0,
                        step=1.0
                    )
                )

                sale_date = (
                    st.date_input(
                        "Sale Date"
                    )
                )

                sale_submit = (
                    st.form_submit_button(
                        "Record Sale",
                        use_container_width=True,
                        type="primary"
                    )
                )

            if sale_submit:

                try:

                    sale_id = (
                        record_sale(
                            product_id=
                                sale_product,

                            quantity_sold=
                                sale_quantity,

                            sale_date=
                                sale_date
                        )
                    )

                    st.success(
                        (
                            f"✅ Sale recorded successfully. "
                            f"Sale: {sale_id}"
                        )
                    )

                    st.cache_data.clear()

                    st.rerun()

                except Exception as error:

                    st.error(
                        str(
                            error
                        )
                    )


    # ========================================================
    # CURRENT INVENTORY


    with inventory_tab4:

        st.subheader(
            "Current Live Inventory"
        )

        snapshot = (
            get_live_inventory_snapshot()
        )

        if snapshot.empty:

            st.info(
                (
                    "No live stock has been entered yet. "
                    "Add a new product or restock an "
                    "existing product."
                )
            )

        else:

            total_units = float(
                snapshot[
                    "current_quantity"
                ]
                .sum()
            )

            c1, c2, c3 = (
                st.columns(3)
            )

            c1.metric(
                "Active Products",
                f"{snapshot['product_id'].nunique():,}"
            )

            c2.metric(
                "Current Units",
                f"{total_units:,.0f}"
            )

            expiring_soon = (
                snapshot[
                    snapshot[
                        "days_to_expiry"
                    ]
                    .between(
                        0,
                        3
                    )
                ]
            )

            c3.metric(
                "Expiring ≤ 3 Days",
                f"{len(expiring_soon):,}"
            )

            st.dataframe(
                snapshot,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader(
                "Inspect Product Batches"
            )

            inspect_product = (
                st.selectbox(
                    "Select Product",
                    snapshot[
                        "product_id"
                    ]
                    .astype(str)
                    .tolist(),
                    key=
                        "inspect_product"
                )
            )

            batches = (
                get_product_batches(
                    inspect_product
                )
            )

            st.dataframe(
                batches,
                use_container_width=True,
                hide_index=True
            )