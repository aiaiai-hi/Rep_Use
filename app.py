# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from io import BytesIO

# Поиск по полям
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Линейность
from pyvis.network import Network
import plotly.graph_objects as go

st.set_page_config(page_title="DWH → Подбор по параметрам, прототип и линейность", layout="wide")

# ============================ УТИЛИТЫ =========================================
def df_to_excel_bytes(df: pd.DataFrame, sheet_name="data") -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()

def download_button_for_df(df, label, filename):
    st.download_button(label=label, data=df_to_excel_bytes(df),
                       file_name=filename, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------- TF-IDF поисковый индекс ----------------
def build_search_text(row, datasets):
    ds = datasets[datasets.dataset_id == row["dataset_id"]].iloc[0]
    return " ".join([
        str(row.get("business_field_name","")),
        str(row.get("business_algorithm","")),
        str(row["column"]),
        " ".join(row.get("tags", [])) if isinstance(row.get("tags"), list) else str(row.get("tags","")),
        f"{row['schema']}.{row['table']}",
        str(ds["name"]),
        str(ds["system"])
    ])

def build_search_index(dataset_fields, datasets):
    df = dataset_fields.copy()
    df["search_text"] = df.apply(lambda r: build_search_text(r, datasets), axis=1)
    vectorizer = TfidfVectorizer(ngram_range=(1,2), analyzer="word", min_df=1)
    tfidf = vectorizer.fit_transform(df["search_text"].astype(str).values)
    return vectorizer, tfidf

def search_fields(query: str, dataset_fields, vectorizer, tfidf, top_k: int = 30):
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, tfidf).ravel()
    idx = np.argsort(sim)[::-1][:top_k]
    return [(dataset_fields.iloc[i]["ref"], float(sim[i])) for i in idx if sim[i] > 0]

# ---------------- Линейность ----------------
def build_lineage_edges(dataset_fields: pd.DataFrame, report_fields: pd.DataFrame):
    """
    dataset(schema.table) -> field(schema.table.column) -> report(report:<id>)
    """
    df = dataset_fields.copy()
    df["dataset"] = df["schema"] + "." + df["table"]
    edges = []
    # dataset -> field
    for _, r in df.iterrows():
        ds = r["dataset"]
        field_ref = f"{r['schema']}.{r['table']}.{r['column']}"
        edges.append((ds, field_ref, "dataset→field"))
    # field -> report
    for _, r in report_fields.iterrows():
        field_ref = r["source_ref"]
        rep = f"report:{r['report_id']}"
        edges.append((field_ref, rep, "field→report"))
    return edges

def pyvis_graph(edges, reports: pd.DataFrame, datasets: pd.DataFrame, height="650px"):
    g = Network(height=height, width="100%", bgcolor="#FFFFFF", font_color="#111111", notebook=False, directed=True)
    g.barnes_hut(gravity=-20000, central_gravity=0.1, spring_length=150, spring_strength=0.01)

    ds_set = set(datasets["name"].tolist())
    report_meta = {f"report:{r.report_id}": r.name for _, r in reports.iterrows()}

    def node_style(n):
        if n in ds_set:             # dataset
            return dict(color="#1f77b4", shape="box")
        if n.startswith("report:"): # report
            return dict(color="#2ca02c", shape="box")
        if n.count(".") == 2:       # field
            return dict(color="#ff7f0e", shape="ellipse")
        return dict(color="#7f7f7f", shape="dot")

    nodes = set()
    for s, t, lbl in edges:
        for n in (s, t):
            if n not in nodes:
                style = node_style(n)
                title = n
                if n in report_meta:
                    title = f"{n} | {report_meta[n]}"
                label = n.split(".")[-1] if n.count(".")>=1 else (report_meta.get(n,n))
                g.add_node(n, label=label, title=title, **style)
                nodes.add(n)
        g.add_edge(s, t, title=lbl, arrows="to")

        # Настройки графа: ВАЖНО — строго JSON, без "var options ="
    g.set_options("""
    {
      "nodes": { "borderWidth": 1, "size": 18 },
      "edges": { "color": { "color": "#B3B3B3" }, "smooth": { "type": "dynamic" } },
      "physics": { "stabilization": true }
    }
    """)

    return g

def sankey_figure(edges):
    nodes = sorted(set([s for s,_,_ in edges] + [t for _,t,_ in edges]))
    idx = {n:i for i,n in enumerate(nodes)}
    source = [idx[s] for s,_,_ in edges]
    target = [idx[t] for _,t,_ in edges]
    value  = [1 for _ in edges]
    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(width=0.5), label=nodes),
        link=dict(source=source, target=target, value=value)
    )])
    fig.update_layout(height=650, margin=dict(l=10,r=10,t=10,b=10))
    return fig

def filter_edges_by_report(edges, report_id: int):
    rnode = f"report:{report_id}"
    keep = set([rnode])
    changed = True
    while changed:
        changed = False
        for s,t,_ in edges:
            if t in keep and s not in keep:
                keep.add(s); changed = True
    return [(s,t,l) for s,t,l in edges if s in keep and t in keep]

# ====================== ТЕСТОВЫЕ ДАННЫЕ (можно заменить) ======================
def load_default_frames():
    datasets = pd.DataFrame([
        {"dataset_id":10,"name":"dm.sales_facts","layer":"vitrine","owner":"DWH","sla_minutes":120,"pii_flags":"","quality_score":0.93,"granularity":"txn_day_sku","system":"DWH / Sales Mart"},
        {"dataset_id":11,"name":"dm.customer_facts","layer":"vitrine","owner":"DWH","sla_minutes":1440,"pii_flags":"PII","quality_score":0.88,"granularity":"customer_month","system":"DWH / CRM Mart"},
        {"dataset_id":12,"name":"raw.erp_costs","layer":"raw","owner":"DataOps","sla_minutes":60,"pii_flags":"","quality_score":0.76,"granularity":"sku_day","system":"ERP"},
        {"dataset_id":13,"name":"dm.finance_facts","layer":"vitrine","owner":"DWH","sla_minutes":1440,"pii_flags":"","quality_score":0.86,"granularity":"dept_month","system":"DWH / Finance Mart"},
        {"dataset_id":14,"name":"raw.crm_events","layer":"raw","owner":"MarTech","sla_minutes":30,"pii_flags":"PII","quality_score":0.71,"granularity":"event","system":"CRM"},
        {"dataset_id":15,"name":"dm.geo_dim","layer":"vitrine","owner":"DWH","sla_minutes":1440,"pii_flags":"","quality_score":0.95,"granularity":"region","system":"DWH / Master Data"},
        {"dataset_id":16,"name":"dm.sales_dim","layer":"vitrine","owner":"DWH","sla_minutes":1440,"pii_flags":"","quality_score":0.92,"granularity":"channel","system":"DWH / Master Data"},
        {"dataset_id":17,"name":"dm.customer_dim","layer":"vitrine","owner":"DWH","sla_minutes":1440,"pii_flags":"PII","quality_score":0.90,"granularity":"customer","system":"DWH / Master Data"},
    ])
    reports = pd.DataFrame([
        {"report_id":1,"name":"Продажи по регионам","owner":"BI Team","frequency":"Ежедневно","business_domain":"Sales","is_automated":True,"automation_score":92,"description":"Воронка продаж и выручка по регионам и каналам."},
        {"report_id":2,"name":"Маржинальность SKU (ручной)","owner":"Finance","frequency":"Еженедельно","business_domain":"Finance","is_automated":False,"automation_score":58,"description":"Ручной excel по марже и себестоимости на уровне SKU."},
        {"report_id":3,"name":"Churn дашборд","owner":"CRM","frequency":"Ежемесячно","business_domain":"CRM","is_automated":True,"automation_score":81,"description":"Отток клиентов, ретеншн и сегменты."},
        {"report_id":4,"name":"План/Факт Доходов","owner":"FP&A","frequency":"Ежемесячно","business_domain":"Finance","is_automated":True,"automation_score":76,"description":"Свод доходов против бюджета по направлениям."},
    ])
    report_fields = pd.DataFrame([
        {"report_id":1,"business_field_name":"Выручка","business_algorithm":"SUM(price*qty) по дню/sku","source_ref":"dm.sales_facts.revenue","is_from_vitrine":True},
        {"report_id":1,"business_field_name":"Канал продаж","business_algorithm":"Справочник каналов","source_ref":"dm.sales_dim.channel","is_from_vitrine":True},
        {"report_id":1,"business_field_name":"Регион","business_algorithm":"Справочник географий","source_ref":"dm.geo_dim.region_name","is_from_vitrine":True},
        {"report_id":2,"business_field_name":"SKU","business_algorithm":"Идентификатор товара (ERP)","source_ref":"raw.erp_costs.sku","is_from_vitrine":False},
        {"report_id":2,"business_field_name":"Себестоимость","business_algorithm":"Сумма прямых затрат (ERP)","source_ref":"raw.erp_costs.cogs","is_from_vitrine":False},
        {"report_id":2,"business_field_name":"Цена продажи","business_algorithm":"Цена из фактов продаж","source_ref":"dm.sales_facts.price","is_from_vitrine":True},
        {"report_id":3,"business_field_name":"Клиент","business_algorithm":"Ключ клиента (Master)","source_ref":"dm.customer_dim.customer_id","is_from_vitrine":True},
        {"report_id":3,"business_field_name":"Статус оттока","business_algorithm":"Флаг churn (логика CRM)","source_ref":"dm.customer_facts.churn_flag","is_from_vitrine":True},
        {"report_id":3,"business_field_name":"Дата последней покупки","business_algorithm":"Max(order_dt) по клиенту","source_ref":"dm.customer_facts.last_purchase_dt","is_from_vitrine":True},
        {"report_id":4,"business_field_name":"Доход факт","business_algorithm":"Факт доходов за период","source_ref":"dm.finance_facts.revenue_actual","is_from_vitrine":True},
        {"report_id":4,"business_field_name":"Доход план","business_algorithm":"Бюджетные значения доходов","source_ref":"dm.finance_facts.revenue_budget","is_from_vitrine":True},
    ])
    dataset_fields = pd.DataFrame([
        {"dataset_id":10,"schema":"dm","table":"sales_facts","column":"revenue","dtype":"decimal","completeness":0.99,"uniqueness":0.95,"tags":["выручка","доход","оборот","revenue","sales"],"business_field_name":"Выручка","business_algorithm":"SUM(price*qty)"},
        {"dataset_id":10,"schema":"dm","table":"sales_facts","column":"price","dtype":"decimal","completeness":0.98,"uniqueness":0.92,"tags":["цена","price","стоимость продажи"],"business_field_name":"Цена продажи","business_algorithm":"price по транзакции"},
        {"dataset_id":10,"schema":"dm","table":"sales_facts","column":"sku_id","dtype":"string","completeness":0.97,"uniqueness":0.80,"tags":["sku","товар","артикул"],"business_field_name":"SKU","business_algorithm":"Ключ SKU"},
        {"dataset_id":10,"schema":"dm","table":"sales_facts","column":"channel","dtype":"string","completeness":0.98,"uniqueness":0.70,"tags":["канал","онлайн","оффлайн","розница","ecom"],"business_field_name":"Канал продаж","business_algorithm":"Справочник каналов"},
        {"dataset_id":10,"schema":"dm","table":"sales_facts","column":"region_id","dtype":"int","completeness":0.98,"uniqueness":0.60,"tags":["регион","гео","область"],"business_field_name":"Регион ID","business_algorithm":"Ссылка на geo_dim"},
        {"dataset_id":11,"schema":"dm","table":"customer_facts","column":"churn_flag","dtype":"bool","completeness":0.97,"uniqueness":1.00,"tags":["отток","churn","ушёл","удержание"],"business_field_name":"Статус оттока","business_algorithm":"ML/правило churn"},
        {"dataset_id":11,"schema":"dm","table":"customer_facts","column":"last_purchase_dt","dtype":"date","completeness":0.96,"uniqueness":0.90,"tags":["последняя покупка","recency","lrp"],"business_field_name":"Дата последней покупки","business_algorithm":"Max(order_dt)"},
        {"dataset_id":13,"schema":"dm","table":"finance_facts","column":"revenue_actual","dtype":"decimal","completeness":0.98,"uniqueness":0.95,"tags":["доход факт","факт","actual","выручка"],"business_field_name":"Доход факт","business_algorithm":"Фактические доходы"},
        {"dataset_id":13,"schema":"dm","table":"finance_facts","column":"revenue_budget","dtype":"decimal","completeness":0.98,"uniqueness":0.95,"tags":["план доход","бюджет","budget"],"business_field_name":"Доход план","business_algorithm":"Бюджет доходов"},
        {"dataset_id":15,"schema":"dm","table":"geo_dim","column":"region_name","dtype":"string","completeness":0.99,"uniqueness":0.95,"tags":["регион","география","регион название"],"business_field_name":"Регион","business_algorithm":"Справочник географий"},
        {"dataset_id":16,"schema":"dm","table":"sales_dim","column":"channel","dtype":"string","completeness":0.99,"uniqueness":0.95,"tags":["канал","канал продаж","розница","marketplace"],"business_field_name":"Канал продаж","business_algorithm":"Справочник каналов"},
        {"dataset_id":17,"schema":"dm","table":"customer_dim","column":"customer_id","dtype":"string","completeness":0.99,"uniqueness":1.00,"tags":["клиент","customer","ид клиента"],"business_field_name":"Клиент","business_algorithm":"Master ID"},
        {"dataset_id":12,"schema":"raw","table":"erp_costs","column":"cogs","dtype":"decimal","completeness":0.92,"uniqueness":0.90,"tags":["себестоимость","cogs","затраты"],"business_field_name":"Себестоимость","business_algorithm":"ERP COGS"},
        {"dataset_id":12,"schema":"raw","table":"erp_costs","column":"sku","dtype":"string","completeness":0.94,"uniqueness":0.80,"tags":["sku","товар","артикул"],"business_field_name":"SKU","business_algorithm":"ERP SKU"},
        {"dataset_id":14,"schema":"raw","table":"crm_events","column":"event_type","dtype":"string","completeness":0.91,"uniqueness":0.65,"tags":["событие","email","push","кампания"],"business_field_name":"Тип события","business_algorithm":"CRM event type"},
    ])
    return datasets, reports, report_fields, dataset_fields

# ============================ SESSION STATE ===================================
if "datasets" not in st.session_state:
    st.session_state.datasets, st.session_state.reports, st.session_state.report_fields, st.session_state.dataset_fields = load_default_frames()
if "selected_refs" not in st.session_state:
    st.session_state.selected_refs = []

datasets = st.session_state.datasets
reports = st.session_state.reports
report_fields = st.session_state.report_fields
dataset_fields = st.session_state.dataset_fields

# Служебные поля
dataset_fields["ref"] = dataset_fields["schema"] + "." + dataset_fields["table"] + "." + dataset_fields["column"]
ref_to_dataset = {r["ref"]: r["dataset_id"] for _, r in dataset_fields.iterrows()}

# Строим/перестраиваем поисковый индекс
if "vectorizer" not in st.session_state or "tfidf" not in st.session_state:
    st.session_state.vectorizer, st.session_state.tfidf = build_search_index(dataset_fields, datasets)

vectorizer = st.session_state.vectorizer
tfidf = st.session_state.tfidf

# =============================== НАВИГАЦИЯ ====================================
page = st.sidebar.radio("Страницы", ["Главная", "Импорт/Экспорт"])

# ================================== ГЛАВНАЯ ===================================
if page == "Главная":
    st.title("Каталог данных и отчётов → Подбор по параметрам, прототип и линейность")
    st.caption("Подберите поля и соберите прототип отчёта, не зная заранее схемы и таблицы.")

    # Табы
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔎 Подбор по параметрам",
        "🧱 Витрины",
        "📋 Реестр отчётов",
        "🗂️ Атрибуты отчёта",
        "🧭 Линейность"
    ])

    # ------------------- TAB 1: Подбор по параметрам -------------------------
    with tab1:
        st.subheader("Опишите что вы хотите")
        st.caption("Введите названия показателей, измерений, систем или таблиц (например: «выручка», «канал продаж», «ERP себестоимость»).")

        q = st.text_input("Поиск по полям (семантический TF-IDF)", placeholder="Например: выручка, канал продаж, регион, себестоимость ...")
        cols = st.columns([3,1])
        with cols[0]:
            if q:
                results = search_fields(q, dataset_fields, vectorizer, tfidf, top_k=50)
                if results:
                    st.write("Найденные поля:")
                    for ref, sc in results:
                        ds_id = ref_to_dataset[ref]
                        ds = datasets[datasets.dataset_id==ds_id].iloc[0]
                        row = dataset_fields.set_index("ref").loc[ref]
                        add_key = f"add_{ref}"
                        with st.container():
                            c1, c2, c3, c4, c5 = st.columns([3,2,1,1,1])
                            c1.markdown(f"**{row.get('business_field_name','')}**  \n`{ref}`")
                            c2.markdown(f"Система: **{ds['system']}**  \nНабор: `{ds['name']}`")
                            c3.markdown(f"Слой: `{ds['layer']}`")
                            c4.markdown(f"DQ: **{row['completeness']:.2f}**")
                            c5.markdown(f"score: {sc:.2f}")
                            if st.button("Добавить в прототип", key=add_key):
                                if ref not in st.session_state.selected_refs:
                                    st.session_state.selected_refs.append(ref)
                else:
                    st.info("Ничего не найдено. Попробуйте другие формулировки.")
            else:
                st.caption("Результаты появятся после ввода поискового запроса.")

        with cols[1]:
            if st.button("Очистить выбранные"):
                st.session_state.selected_refs = []

        st.markdown("---")
        st.markdown("### Прототип отчёта")
        if not st.session_state.selected_refs:
            st.info("Добавляйте поля из результатов поиска — они появятся в таблице ниже.")
        else:
            # Таблица прототипа с требуемыми колонками
            rows = []
            for i, ref in enumerate(st.session_state.selected_refs, start=1):
                ds_id = ref_to_dataset[ref]
                ds = datasets[datasets.dataset_id==ds_id].iloc[0]
                rf = report_fields[report_fields["source_ref"]==ref]
                used_ids = rf["report_id"].tolist()
                used_names = reports[reports["report_id"].isin(used_ids)]["name"].tolist()
                row = dataset_fields.set_index("ref").loc[ref]
                rows.append({
                    "№": i,
                    "Бизнес-поле": row.get("business_field_name",""),
                    "Бизнес-алгоритм": row.get("business_algorithm",""),
                    "Источник (schema.table.column)": ref,
                    "Связь с информационной системой": ds["system"],
                    "В какую таблицу входит": f"{row['schema']}.{row['table']}",
                    "В каких отчётах есть (ID)": ", ".join(map(str, used_ids)) if used_ids else "—",
                    "Названия отчётов": ", ".join(used_names) if used_names else "—",
                })
            df_proto = pd.DataFrame(rows)
            st.dataframe(df_proto, use_container_width=True, height=360)
            download_button_for_df(df_proto, "⬇️ Скачать прототип (Excel)", "prototype.xlsx")

    # ------------------- TAB 2: Витрины --------------------------------------
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Витрина (dm.*)**")
            st.dataframe(datasets.query("layer=='vitrine'")[["name","system","owner","sla_minutes","pii_flags","quality_score","granularity"]]
                         .rename(columns={"name":"Набор","system":"Система","owner":"Владелец","sla_minutes":"SLA (мин)","pii_flags":"PII","quality_score":"Качество","granularity":"Гранулярность"}),
                         use_container_width=True, height=260)
        with c2:
            st.write("**Источники (RAW/Source)**")
            st.dataframe(datasets.query("layer!='vitrine'")[["name","system","owner","sla_minutes","pii_flags","quality_score","granularity"]]
                         .rename(columns={"name":"Набор","system":"Система","owner":"Владелец","sla_minutes":"SLA (мин)","pii_flags":"PII","quality_score":"Качество","granularity":"Гранулярность"}),
                         use_container_width=True, height=260)

        st.markdown("### Поля")
        df_fields = dataset_fields.copy()
        df_fields.insert(0, "№", range(1, len(df_fields)+1))
        # Колонки: после нумерации — Бизнес-поле, Бизнес-алгоритм, затем схема/таблица/поле и пр.
        df_fields = df_fields[["№","business_field_name","business_algorithm","schema","table","column","dtype","completeness","uniqueness","tags"]]
        df_fields = df_fields.rename(columns={
            "business_field_name":"Бизнес-поле",
            "business_algorithm":"Бизнес-алгоритм",
            "schema":"Схема","table":"Таблица","column":"Поле","dtype":"Тип",
            "completeness":"Полнота","uniqueness":"Уникальность","tags":"Теги"
        })
        st.dataframe(df_fields, use_container_width=True, height=360)

    # ------------------- TAB 3: Реестр отчётов (поисковый фильтр) -------------
    with tab3:
        st.write("### Реестр отчётов")
        query = st.text_input("Фильтр по наименованию/владельцу/домену", "")
        show = reports.copy()
        if query:
            ql = query.lower()
            mask = (
                show["name"].str.lower().str.contains(ql) |
                show["owner"].str.lower().str.contains(ql) |
                show["business_domain"].str.lower().str.contains(ql)
            )
            show = show[mask]
        grid = show[["name","owner","business_domain","frequency","is_automated","automation_score","description"]].rename(
            columns={"name":"Наименование","owner":"Владелец","business_domain":"Домен","frequency":"Частота","is_automated":"Авто?","automation_score":"Скор","description":"Описание"})
        st.dataframe(grid, use_container_width=True, height=300)

    # ------------------- TAB 4: Атрибуты отчёта (поиск + выгрузка) ------------
    with tab4:
        selected = st.selectbox("Выберите отчёт", options=reports["name"].tolist())
        rep = reports[reports["name"]==selected].iloc[0]
        rid = rep["report_id"]
        st.write(f"**Владелец:** {rep['owner']}  ·  **Частота:** {rep['frequency']}  ·  **Статус:** {'Автоматизирован' if rep['is_automated'] else 'Ручной'}")
        st.caption(rep["description"])

        rf = report_fields[report_fields["report_id"]==rid].copy()
        rf["Наименование отчёта"] = rep["name"]
        rf = rf.rename(columns={
            "report_id":"Код отчёта",
            "business_field_name":"Бизнес-поле",
            "business_algorithm":"Бизнес-алгоритм",
            "source_ref":"Источник (schema.table.column)",
            "is_from_vitrine":"Из витрины?"
        })
        # Порядок: Бизнес-алгоритм → Код отчёта → Наименование отчёта → ...
        rf = rf[["Бизнес-поле","Бизнес-алгоритм","Код отчёта","Наименование отчёта","Источник (schema.table.column)","Из витрины?"]]

        attr_filter = st.text_input("Фильтр по атрибутам (поле/алгоритм/источник)")
        if attr_filter:
            ql = attr_filter.lower()
            mask = (
                rf["Бизнес-поле"].str.lower().str.contains(ql) |
                rf["Бизнес-алгоритм"].str.lower().str.contains(ql) |
                rf["Источник (schema.table.column)"].str.lower().str.contains(ql)
            )
            rf_view = rf[mask]
        else:
            rf_view = rf

        st.dataframe(rf_view, use_container_width=True, height=280)
        download_button_for_df(rf_view, "⬇️ Скачать атрибуты отчёта (Excel)", f"report_{rid}_attrs.xlsx")

    # ------------------- TAB 5: Линейность -----------------------------------
    with tab5:
        st.subheader("Data Lineage")

        mode = st.radio("Режим", ["Глобально", "По отчёту"], horizontal=True)
        if mode == "По отчёту":
            rep_name = st.selectbox("Выберите отчёт", options=reports["name"].tolist())
            rid = int(reports[reports["name"]==rep_name].iloc[0]["report_id"])
        else:
            rid = None

        edges_all = build_lineage_edges(dataset_fields, report_fields)
        edges = filter_edges_by_report(edges_all, rid) if rid else edges_all

        viz = st.radio("Тип визуализации", ["Граф (интерактивный)", "Sankey"], horizontal=True)

        if viz == "Граф (интерактивный)":
            net = pyvis_graph(edges, reports, datasets, height="650px")
            html = net.generate_html(notebook=False)
            components.html(html, height=680, scrolling=True)
            st.download_button(
                "⬇️ Скачать граф (HTML)",
                data=html.encode("utf-8"),
                file_name=f"lineage_{'report_'+str(rid) if rid else 'global'}.html",
                mime="text/html"
            )
        else:
            fig = sankey_figure(edges)
            st.plotly_chart(fig, use_container_width=True)

# ============================== ИМПОРТ / ЭКСПОРТ ==============================
else:
    st.title("Импорт/Экспорт")
    st.caption("Загрузите Excel-файлы, чтобы заменить тестовые данные. Также можно скачать шаблоны для заполнения.")

    st.markdown("#### Шаблоны для выгрузки")
    template_datasets = pd.DataFrame([{
        "dataset_id":"","name":"","layer":"","owner":"","sla_minutes":"","pii_flags":"","quality_score":"","granularity":"","system":""
    }])
    template_dataset_fields = pd.DataFrame([{
        "dataset_id":"","schema":"","table":"","column":"","dtype":"",
        "completeness":"","uniqueness":"","tags":"список_через_запятую",
        "business_field_name":"","business_algorithm":""
    }])
    template_reports = pd.DataFrame([{
        "report_id":"","name":"","owner":"","frequency":"","business_domain":"",
        "is_automated":"","automation_score":"","description":""
    }])
    template_report_fields = pd.DataFrame([{
        "report_id":"","business_field_name":"","business_algorithm":"",
        "source_ref":"","is_from_vitrine":""
    }])

    c1, c2, c3, c4 = st.columns(4)
    with c1: download_button_for_df(template_datasets, "⬇️ Шаблон: datasets.xlsx", "datasets_template.xlsx")
    with c2: download_button_for_df(template_dataset_fields, "⬇️ Шаблон: dataset_fields.xlsx", "dataset_fields_template.xlsx")
    with c3: download_button_for_df(template_reports, "⬇️ Шаблон: reports.xlsx", "reports_template.xlsx")
    with c4: download_button_for_df(template_report_fields, "⬇️ Шаблон: report_fields.xlsx", "report_fields_template.xlsx")

    st.markdown("---")
    st.markdown("#### Импорт Excel")
    st.caption("Ожидаются отдельные файлы с соответствующими столбцами. Теги можно указывать через запятую.")

    up1 = st.file_uploader("Загрузите datasets.xlsx", type=["xlsx"])
    up2 = st.file_uploader("Загрузите dataset_fields.xlsx", type=["xlsx"])
    up3 = st.file_uploader("Загрузите reports.xlsx", type=["xlsx"])
    up4 = st.file_uploader("Загрузите report_fields.xlsx", type=["xlsx"])

    if st.button("Заменить тестовые данные на загруженные"):
        try:
            if up1: st.session_state.datasets = pd.read_excel(up1)
            if up2:
                df = pd.read_excel(up2)
                if "tags" in df.columns:
                    df["tags"] = df["tags"].apply(lambda x: [t.strip() for t in str(x).split(",")] if pd.notna(x) else [])
                st.session_state.dataset_fields = df
            if up3: st.session_state.reports = pd.read_excel(up3)
            if up4: st.session_state.report_fields = pd.read_excel(up4)

            # Пересоберём служебные поля/индекс
            st.session_state.dataset_fields["ref"] = (
                st.session_state.dataset_fields["schema"] + "." +
                st.session_state.dataset_fields["table"] + "." +
                st.session_state.dataset_fields["column"]
            )
            st.session_state.vectorizer, st.session_state.tfidf = build_search_index(
                st.session_state.dataset_fields, st.session_state.datasets
            )
            st.success("Данные обновлены. Перейдите на страницу «Главная».")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")