# app.py
import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="DWH → Подбор по параметрам и прототип отчёта", layout="wide")

# ------------------------------ ТЕСТОВЫЕ ДАННЫЕ -----------------------------------
# Добавили колонку 'system' для связи с ИС
datasets = pd.DataFrame([
    {"dataset_id": 10, "name": "dm.sales_facts",    "layer": "vitrine", "owner": "DWH",     "sla_minutes": 120,  "pii_flags": "",     "quality_score": 0.93, "granularity": "txn_day_sku", "system": "DWH / Sales Mart"},
    {"dataset_id": 11, "name": "dm.customer_facts", "layer": "vitrine", "owner": "DWH",     "sla_minutes": 1440, "pii_flags": "PII",  "quality_score": 0.88, "granularity": "customer_month","system": "DWH / CRM Mart"},
    {"dataset_id": 12, "name": "raw.erp_costs",     "layer": "raw",     "owner": "DataOps", "sla_minutes": 60,   "pii_flags": "",     "quality_score": 0.76, "granularity": "sku_day",      "system": "ERP"},
    {"dataset_id": 13, "name": "dm.finance_facts",  "layer": "vitrine", "owner": "DWH",     "sla_minutes": 1440, "pii_flags": "",     "quality_score": 0.86, "granularity": "dept_month",   "system": "DWH / Finance Mart"},
    {"dataset_id": 14, "name": "raw.crm_events",    "layer": "raw",     "owner": "MarTech", "sla_minutes": 30,   "pii_flags": "PII",  "quality_score": 0.71, "granularity": "event",        "system": "CRM"},
    {"dataset_id": 15, "name": "dm.geo_dim",        "layer": "vitrine", "owner": "DWH",     "sla_minutes": 1440, "pii_flags": "",     "quality_score": 0.95, "granularity": "region",       "system": "DWH / Master Data"},
    {"dataset_id": 16, "name": "dm.sales_dim",      "layer": "vitrine", "owner": "DWH",     "sla_minutes": 1440, "pii_flags": "",     "quality_score": 0.92, "granularity": "channel",      "system": "DWH / Master Data"},
    {"dataset_id": 17, "name": "dm.customer_dim",   "layer": "vitrine", "owner": "DWH",     "sla_minutes": 1440, "pii_flags": "PII",  "quality_score": 0.90, "granularity": "customer",     "system": "DWH / Master Data"},
])

reports = pd.DataFrame([
    {"report_id": 1, "name": "Продажи по регионам",          "owner": "BI Team", "frequency": "Ежедневно",   "business_domain": "Sales",   "is_automated": True,  "automation_score": 92, "description": "Воронка продаж и выручка по регионам и каналам."},
    {"report_id": 2, "name": "Маржинальность SKU (ручной)",  "owner": "Finance", "frequency": "Еженедельно", "business_domain": "Finance", "is_automated": False, "automation_score": 58, "description": "Ручной excel по марже и себестоимости на уровне SKU."},
    {"report_id": 3, "name": "Churn дашборд",                 "owner": "CRM",     "frequency": "Ежемесячно",  "business_domain": "CRM",     "is_automated": True,  "automation_score": 81, "description": "Отчёт по оттоку клиентов, ретеншн и сегменты."},
    {"report_id": 4, "name": "План/Факт Доходов",             "owner": "FP&A",    "frequency": "Ежемесячно",  "business_domain": "Finance", "is_automated": True,  "automation_score": 76, "description": "Свод доходов против бюджета по направлениям."},
])

report_fields = pd.DataFrame([
    {"report_id": 1, "business_field_name": "Выручка",                 "source_ref": "dm.sales_facts.revenue",            "is_from_vitrine": True},
    {"report_id": 1, "business_field_name": "Канал продаж",            "source_ref": "dm.sales_dim.channel",              "is_from_vitrine": True},
    {"report_id": 1, "business_field_name": "Регион",                  "source_ref": "dm.geo_dim.region_name",            "is_from_vitrine": True},
    {"report_id": 2, "business_field_name": "SKU",                     "source_ref": "raw.erp_costs.sku",                 "is_from_vitrine": False},
    {"report_id": 2, "business_field_name": "Себестоимость",           "source_ref": "raw.erp_costs.cogs",                "is_from_vitrine": False},
    {"report_id": 2, "business_field_name": "Цена продажи",            "source_ref": "dm.sales_facts.price",              "is_from_vitrine": True},
    {"report_id": 3, "business_field_name": "Клиент",                  "source_ref": "dm.customer_dim.customer_id",       "is_from_vitrine": True},
    {"report_id": 3, "business_field_name": "Статус оттока",           "source_ref": "dm.customer_facts.churn_flag",      "is_from_vitrine": True},
    {"report_id": 3, "business_field_name": "Дата последней покупки",  "source_ref": "dm.customer_facts.last_purchase_dt","is_from_vitrine": True},
    {"report_id": 4, "business_field_name": "Доход факт",              "source_ref": "dm.finance_facts.revenue_actual",   "is_from_vitrine": True},
    {"report_id": 4, "business_field_name": "Доход план",              "source_ref": "dm.finance_facts.revenue_budget",   "is_from_vitrine": True},
])

dataset_fields = pd.DataFrame([
    {"dataset_id": 10, "schema": "dm",  "table": "sales_facts",    "column": "revenue",           "dtype": "decimal", "completeness": 0.99, "uniqueness": 0.95, "tags": ["выручка","доход","оборот","revenue","sales"]},
    {"dataset_id": 10, "schema": "dm",  "table": "sales_facts",    "column": "price",             "dtype": "decimal", "completeness": 0.98, "uniqueness": 0.92, "tags": ["цена","price","стоимость продажи"]},
    {"dataset_id": 10, "schema": "dm",  "table": "sales_facts",    "column": "sku_id",            "dtype": "string",  "completeness": 0.97, "uniqueness": 0.80, "tags": ["sku","товар","артикул"]},
    {"dataset_id": 10, "schema": "dm",  "table": "sales_facts",    "column": "channel",           "dtype": "string",  "completeness": 0.98, "uniqueness": 0.70, "tags": ["канал","онлайн","оффлайн","розница","ecom"]},
    {"dataset_id": 10, "schema": "dm",  "table": "sales_facts",    "column": "region_id",         "dtype": "int",     "completeness": 0.98, "uniqueness": 0.60, "tags": ["регион","гео","область"]},
    {"dataset_id": 11, "schema": "dm",  "table": "customer_facts", "column": "churn_flag",        "dtype": "bool",    "completeness": 0.97, "uniqueness": 1.00, "tags": ["отток","churn","ушёл","удержание"]},
    {"dataset_id": 11, "schema": "dm",  "table": "customer_facts", "column": "last_purchase_dt",  "dtype": "date",    "completeness": 0.96, "uniqueness": 0.90, "tags": ["последняя покупка","recency","lrp"]},
    {"dataset_id": 13, "schema": "dm",  "table": "finance_facts",  "column": "revenue_actual",    "dtype": "decimal", "completeness": 0.98, "uniqueness": 0.95, "tags": ["доход факт","факт","actual","выручка"]},
    {"dataset_id": 13, "schema": "dm",  "table": "finance_facts",  "column": "revenue_budget",    "dtype": "decimal", "completeness": 0.98, "uniqueness": 0.95, "tags": ["план доход","бюджет","budget"]},
    {"dataset_id": 15, "schema": "dm",  "table": "geo_dim",        "column": "region_name",       "dtype": "string",  "completeness": 0.99, "uniqueness": 0.95, "tags": ["регион","география","регион название"]},
    {"dataset_id": 16, "schema": "dm",  "table": "sales_dim",      "column": "channel",           "dtype": "string",  "completeness": 0.99, "uniqueness": 0.95, "tags": ["канал","канал продаж","розница","marketplace"]},
    {"dataset_id": 17, "schema": "dm",  "table": "customer_dim",   "column": "customer_id",       "dtype": "string",  "completeness": 0.99, "uniqueness": 1.00, "tags": ["клиент","customer","ид клиента"]},
    {"dataset_id": 12, "schema": "raw", "table": "erp_costs",      "column": "cogs",              "dtype": "decimal", "completeness": 0.92, "uniqueness": 0.90, "tags": ["себестоимость","cogs","затраты"]},
    {"dataset_id": 12, "schema": "raw", "table": "erp_costs",      "column": "sku",               "dtype": "string",  "completeness": 0.94, "uniqueness": 0.80, "tags": ["sku","товар","артикул"]},
    {"dataset_id": 14, "schema": "raw", "table": "crm_events",     "column": "event_type",        "dtype": "string",  "completeness": 0.91, "uniqueness": 0.65, "tags": ["событие","email","push","кампания"]},
])

dataset_fields["ref"] = dataset_fields["schema"] + "." + dataset_fields["table"] + "." + dataset_fields["column"]
ref_to_dataset = {r["ref"]: r["dataset_id"] for _, r in dataset_fields.iterrows()}

# Индексация: текст для поиска = column + tags + dataset name + system
def build_corpus_row(row):
    ds = datasets[datasets.dataset_id == row["dataset_id"]].iloc[0]
    parts = [
        row["column"],
        " ".join(row["tags"]),
        f"{row['schema']}.{row['table']}",
        ds["name"],
        ds["system"]
    ]
    return " ".join([str(x) for x in parts if x])

dataset_fields["search_text"] = dataset_fields.apply(build_corpus_row, axis=1)

# TF-IDF индекс
vectorizer = TfidfVectorizer(ngram_range=(1,2), analyzer="word", min_df=1)
tfidf = vectorizer.fit_transform(dataset_fields["search_text"].values)

# ------------------------------ ХЕЛПЕРЫ -----------------------------------
def search_fields(query: str, top_k: int = 20):
    """Возвращает top_k полей с косинусной близостью по TF-IDF."""
    if not query.strip():
        return []
    q_vec = vectorizer.transform([query])
    sim = cosine_similarity(q_vec, tfidf).ravel()
    idx = np.argsort(sim)[::-1][:top_k]
    results = []
    for i in idx:
        results.append((dataset_fields.iloc[i]["ref"], float(sim[i])))
    return [r for r in results if r[1] > 0]

def status_label(score):
    if score >= 85: return "✅ Готово"
    if score >= 60: return "🟡 Частично"
    return "🔴 Требуется доработка"

def feasibility_score(found_refs, total_refs, allow_vitrine=True):
    if total_refs == 0:
        return 0, {}
    coverage = len(found_refs) / total_refs
    freshness = 0.8 if allow_vitrine else 0.9
    quality = float(np.mean([dataset_fields.set_index("ref").loc[r, "completeness"] for r in found_refs])) if found_refs else 0.5
    access = 0.9
    vit_share = sum(1 for r in found_refs if r.startswith("dm.")) / max(1,len(found_refs))
    reuse = 1.0 if vit_share >= 0.7 else 0.4
    score = (coverage*0.40 + freshness*0.20 + quality*0.15 + access*0.15 + reuse*0.10) * 100
    return round(score), {
        "Покрытие полей": round(coverage*100),
        "Свежесть/SLA": round(freshness*100),
        "Качество данных": round(quality*100),
        "Доступ": round(access*100),
        "Переиспользование": round(reuse*100),
    }

def reports_for_ref(ref: str):
    used = report_fields[report_fields["source_ref"] == ref]["report_id"].tolist()
    names = reports[reports["report_id"].isin(used)]["name"].tolist()
    return used, names

# ------------------------------ СОСТОЯНИЕ -----------------------------------
if "selected_refs" not in st.session_state:
    st.session_state.selected_refs = []

# ------------------------------ HEADER ---------------------------------------
st.title("Каталог данных и отчётов → Подбор по параметрам и сборка прототипа")
st.caption("Подберите поля и соберите прототип отчёта, не зная заранее схемы и таблицы.")

# ------------------------------ ТАБЫ -----------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔎 Подбор по параметрам", "🧱 Данные", "📊 Отчёты", "🧪 Проверка пригодности"])

# ------------------------------ TAB 1: ПОДБОР ПО ПАРАМЕТРАМ ------------------
with tab1:
    st.subheader("Опишите что вы хотите")
    st.caption("Начните вводить названия показателей, измерений, систем или таблиц (например: «выручка», «канал продаж», «ERP себестоимость», «finance facts»).")
    q = st.text_input("Поиск по полям (TF-IDF индекс)", placeholder="Например: выручка по каналам и регионам, себестоимость SKU ...")

    cols = st.columns([3,1])
    with cols[0]:
        if q:
            results = search_fields(q, top_k=30)
            if results:
                st.write("Найденные поля:")
                for ref, sc in results:
                    ds_id = ref_to_dataset[ref]
                    ds = datasets[datasets.dataset_id==ds_id].iloc[0]
                    meta = dataset_fields.set_index("ref").loc[ref]
                    add_key = f"add_{ref}"
                    with st.container():
                        c1, c2, c3, c4, c5 = st.columns([3,2,1,1,1])
                        c1.markdown(f"**{ref.split('.')[-1]}**  \n`{ref}`")
                        c2.markdown(f"Набор: `{ds['name']}`  \nСистема: **{ds['system']}**")
                        c3.markdown(f"Слой: `{ds['layer']}`")
                        c4.markdown(f"DQ: **{meta['completeness']:.2f}**")
                        c5.markdown(f"score: {sc:.2f}")
                        if st.button("Добавить в прототип", key=add_key):
                            if ref not in st.session_state.selected_refs:
                                st.session_state.selected_refs.append(ref)
            else:
                st.info("Ничего не найдено. Попробуйте другие формулировки или более общий термин.")
        else:
            st.caption("Результаты появятся после ввода поискового запроса.")

    with cols[1]:
        if st.button("Очистить выбранные"):
            st.session_state.selected_refs = []

    st.markdown("---")
    st.markdown("### Прототип отчёта (сборщик)")
    if not st.session_state.selected_refs:
        st.info("Пока ничего не выбрано. Добавляйте поля из результатов поиска — они появятся здесь.")
    else:
        # Построим «вертикальную таблицу» (по строке на поле) с нужными колонками
        rows = []
        for ref in st.session_state.selected_refs:
            ds_id = ref_to_dataset[ref]
            ds = datasets[datasets.dataset_id==ds_id].iloc[0]
            used_ids, used_names = reports_for_ref(ref)
            rows.append({
                "Поле": ref.split(".")[-1],
                "Связь с информационной системой": ds["system"],
                "В какую таблицу входит": f"{ref.split('.')[0]}.{ref.split('.')[1]}",
                "Источник (schema.table.column)": ref,
                "В каких отчётах есть (ID)": ", ".join(map(str, used_ids)) if used_ids else "—",
                "Названия отчётов": ", ".join(used_names) if used_names else "—",
            })
        df_proto = pd.DataFrame(rows)
        st.dataframe(df_proto, use_container_width=True, height=320)

        # Быстрая оценка пригодности выбранного набора
        score, br = feasibility_score(st.session_state.selected_refs, len(st.session_state.selected_refs), allow_vitrine=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Feasibility", f"{score}/100")
        c2.metric("Статус", status_label(score))
        c3.metric("Поля", str(len(st.session_state.selected_refs)))
        st.progress(min(1.0, score/100))
        with st.expander("Детализация оценки"):
            st.write(br)

# ------------------------------ TAB 2: ДАННЫЕ --------------------------------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Витрина (dm.\*)**")
        st.dataframe(datasets.query("layer=='vitrine'")[["name","system","owner","sla_minutes","pii_flags","quality_score","granularity"]]
            .rename(columns={"name":"Набор","system":"Система","owner":"Владелец","sla_minutes":"SLA (мин)","pii_flags":"PII","quality_score":"Качество","granularity":"Гранулярность"}),
            use_container_width=True, height=260)
    with c2:
        st.write("**Источники (RAW/Source)**")
        st.dataframe(datasets.query("layer!='vitrine'")[["name","system","owner","sla_minutes","pii_flags","quality_score","granularity"]]
            .rename(columns={"name":"Набор","system":"Система","owner":"Владелец","sla_minutes":"SLA (мин)","pii_flags":"PII","quality_score":"Качество","granularity":"Гранулярность"}),
            use_container_width=True, height=260)

    st.markdown("### Поля")
    st.dataframe(dataset_fields[["schema","table","column","dtype","completeness","uniqueness","tags"]]
        .rename(columns={"schema":"Схема","table":"Таблица","column":"Поле","dtype":"Тип","completeness":"Полнота","uniqueness":"Уникальность","tags":"Теги"}),
        use_container_width=True, height=320)

# ------------------------------ TAB 3: ОТЧЁТЫ --------------------------------
with tab3:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего отчётов", len(reports))
    col2.metric("Автоматизировано", int(reports["is_automated"].sum()))
    col3.metric("Витрин", datasets.query("layer=='vitrine'").shape[0])
    col4.metric("Источников", datasets.query("layer!='vitrine'").shape[0])

    st.write("### Каталог отчётов")
    show = reports.copy()
    st.dataframe(show[["name","owner","business_domain","frequency","is_automated","automation_score","description"]]
        .rename(columns={"name":"Название","owner":"Владелец","business_domain":"Домен","frequency":"Частота","is_automated":"Авто?","automation_score":"Скор","description":"Описание"}),
        use_container_width=True, height=260)

    st.markdown("### Детали")
    selected = st.selectbox("Выберите отчёт", options=show["name"].tolist())
    rep = show[show["name"]==selected].iloc[0]
    rid = rep["report_id"]
    cols = st.columns(4)
    cols[0].write(f"**Владелец:** {rep['owner']}")
    cols[1].write(f"**Частота:** {rep['frequency']}")
    cols[2].write(f"**Статус:** {'Автоматизирован' if rep['is_automated'] else 'Ручной'}")
    cols[3].write(f"**Готовность:** {status_label(rep['automation_score'])}")
    st.caption(rep["description"])
    rf = report_fields[report_fields["report_id"]==rid]
    st.write("**Поля и источники:**")
    st.dataframe(rf.rename(columns={
        "business_field_name":"Бизнес-поле","source_ref":"Источник (schema.table.column)","is_from_vitrine":"Из витрины?"
    }), use_container_width=True, height=220)

# ------------------------------ TAB 4: ПРОВЕРКА ПРИГОДНОСТИ ------------------
with tab4:
    st.write("Вставьте список полей (`schema.table.column`) или соберите его на вкладке «Подбор по параметрам».")
    req = st.text_area("Требуемые поля", "\n".join(st.session_state.selected_refs) if st.session_state.selected_refs else "dm.sales_facts.revenue\ndm.sales_dim.channel\ndm.geo_dim.region_name")
    allow_v = st.toggle("Можно использовать витрину", value=True, key="allow_v2")
    if st.button("Проверить пригодность"):
        requested = [x.strip() for x in req.split("\n") if x.strip()]
        known_refs = set(dataset_fields["ref"].tolist())
        found = [r for r in requested if r in known_refs]
        score, br = feasibility_score(found, len(requested), allow_vitrine=allow_v)
        g1, g2, g3 = st.columns(3)
        g1.metric("Feasibility", f"{score}/100")
        g2.metric("Статус", status_label(score))
        g3.metric("Покрытие полей", f"{br.get('Покрытие полей',0)}%")
        st.progress(min(1.0, score/100))
        st.subheader("Детализация")
        st.write(br)
        miss = [r for r in requested if r not in found]
        st.write("**Найдено:**", found if found else "—")
        st.write("**Отсутствует:**", miss if miss else "—")
