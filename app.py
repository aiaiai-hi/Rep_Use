# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter

st.set_page_config(page_title="DWH → Report Feasibility", layout="wide")

# ------------------------------ TEST DATA -----------------------------------
# Reports
reports = pd.DataFrame([
    {"report_id": 1, "name": "Продажи по регионам", "owner": "BI Team", "frequency": "Ежедневно",
     "business_domain": "Sales", "is_automated": True, "automation_score": 92,
     "description": "Воронка продаж и выручка по регионам и каналам."},
    {"report_id": 2, "name": "Маржинальность SKU (ручной)", "owner": "Finance", "frequency": "Еженедельно",
     "business_domain": "Finance", "is_automated": False, "automation_score": 58,
     "description": "Ручной excel по марже и себестоимости на уровне SKU."},
    {"report_id": 3, "name": "Churn дашборд", "owner": "CRM", "frequency": "Ежемесячно",
     "business_domain": "CRM", "is_automated": True, "automation_score": 81,
     "description": "Отчёт по оттоку клиентов, ретеншн и сегменты."},
    {"report_id": 4, "name": "План/Факт Доходов", "owner": "FP&A", "frequency": "Ежемесячно",
     "business_domain": "Finance", "is_automated": True, "automation_score": 76,
     "description": "Свод доходов против бюджета по направлениям."},
])

# Report fields (link business fields to physical refs)
report_fields = pd.DataFrame([
    {"report_id": 1, "business_field_name": "Выручка", "source_ref": "dm.sales_facts.revenue", "is_from_vitrine": True},
    {"report_id": 1, "business_field_name": "Канал продаж", "source_ref": "dm.sales_dim.channel", "is_from_vitrine": True},
    {"report_id": 1, "business_field_name": "Регион", "source_ref": "dm.geo_dim.region_name", "is_from_vitrine": True},
    {"report_id": 2, "business_field_name": "SKU", "source_ref": "raw.erp_items.sku", "is_from_vitrine": False},
    {"report_id": 2, "business_field_name": "Себестоимость", "source_ref": "raw.erp_costs.cogs", "is_from_vitrine": False},
    {"report_id": 2, "business_field_name": "Цена продажи", "source_ref": "dm.sales_facts.price", "is_from_vitrine": True},
    {"report_id": 3, "business_field_name": "Клиент", "source_ref": "dm.customer_dim.customer_id", "is_from_vitrine": True},
    {"report_id": 3, "business_field_name": "Статус оттока", "source_ref": "dm.customer_facts.churn_flag", "is_from_vitrine": True},
    {"report_id": 3, "business_field_name": "Дата последней покупки", "source_ref": "dm.customer_facts.last_purchase_dt", "is_from_vitrine": True},
    {"report_id": 4, "business_field_name": "Доход факт", "source_ref": "dm.finance_facts.revenue_actual", "is_from_vitrine": True},
    {"report_id": 4, "business_field_name": "Доход план", "source_ref": "dm.finance_facts.revenue_budget", "is_from_vitrine": True},
])

# Datasets (vitrine & raw)
datasets = pd.DataFrame([
    {"dataset_id": 10, "name": "dm.sales_facts", "layer": "vitrine", "owner": "DWH", "sla_minutes": 120, "pii_flags": "", "quality_score": 0.93, "granularity": "txn_day_sku"},
    {"dataset_id": 11, "name": "dm.customer_facts", "layer": "vitrine", "owner": "DWH", "sla_minutes": 1440, "pii_flags": "PII", "quality_score": 0.88, "granularity": "customer_month"},
    {"dataset_id": 12, "name": "raw.erp_costs", "layer": "raw", "owner": "DataOps", "sla_minutes": 60, "pii_flags": "", "quality_score": 0.76, "granularity": "sku_day"},
    {"dataset_id": 13, "name": "dm.finance_facts", "layer": "vitrine", "owner": "DWH", "sla_minutes": 1440, "pii_flags": "", "quality_score": 0.86, "granularity": "dept_month"},
    {"dataset_id": 14, "name": "raw.crm_events", "layer": "raw", "owner": "MarTech", "sla_minutes": 30, "pii_flags": "PII", "quality_score": 0.71, "granularity": "event"},
    {"dataset_id": 15, "name": "dm.geo_dim", "layer": "vitrine", "owner": "DWH", "sla_minutes": 1440, "pii_flags": "", "quality_score": 0.95, "granularity": "region"},
    {"dataset_id": 16, "name": "dm.sales_dim", "layer": "vitrine", "owner": "DWH", "sla_minutes": 1440, "pii_flags": "", "quality_score": 0.92, "granularity": "channel"},
    {"dataset_id": 17, "name": "dm.customer_dim", "layer": "vitrine", "owner": "DWH", "sla_minutes": 1440, "pii_flags": "PII", "quality_score": 0.90, "granularity": "customer"},
])

# Dataset fields (with simple quality metrics)
dataset_fields = pd.DataFrame([
    # sales_facts
    {"dataset_id": 10, "schema": "dm", "table": "sales_facts", "column": "revenue", "dtype": "decimal", "completeness": 0.99, "uniqueness": 0.95, "tags": ["выручка","доход","оборот","revenue","sales"]},
    {"dataset_id": 10, "schema": "dm", "table": "sales_facts", "column": "price", "dtype": "decimal", "completeness": 0.98, "uniqueness": 0.92, "tags": ["цена","price","стоимость продажи"]},
    {"dataset_id": 10, "schema": "dm", "table": "sales_facts", "column": "sku_id", "dtype": "string", "completeness": 0.97, "uniqueness": 0.80, "tags": ["sku","товар","артикул"]},
    {"dataset_id": 10, "schema": "dm", "table": "sales_facts", "column": "channel", "dtype": "string", "completeness": 0.98, "uniqueness": 0.70, "tags": ["канал","онлайн","оффлайн","розница","ecom"]},
    {"dataset_id": 10, "schema": "dm", "table": "sales_facts", "column": "region_id", "dtype": "int", "completeness": 0.98, "uniqueness": 0.60, "tags": ["регион","гео","область"]},

    # customer_facts
    {"dataset_id": 11, "schema": "dm", "table": "customer_facts", "column": "churn_flag", "dtype": "bool", "completeness": 0.97, "uniqueness": 1.00, "tags": ["отток","churn","ушёл","удержание"]},
    {"dataset_id": 11, "schema": "dm", "table": "customer_facts", "column": "last_purchase_dt", "dtype": "date", "completeness": 0.96, "uniqueness": 0.90, "tags": ["последняя покупка","recency","lrp"]},

    # finance_facts
    {"dataset_id": 13, "schema": "dm", "table": "finance_facts", "column": "revenue_actual", "dtype": "decimal", "completeness": 0.98, "uniqueness": 0.95, "tags": ["доход факт","факт","actual","выручка"]},
    {"dataset_id": 13, "schema": "dm", "table": "finance_facts", "column": "revenue_budget", "dtype": "decimal", "completeness": 0.98, "uniqueness": 0.95, "tags": ["план доход","бюджет","budget"]},

    # geo_dim
    {"dataset_id": 15, "schema": "dm", "table": "geo_dim", "column": "region_name", "dtype": "string", "completeness": 0.99, "uniqueness": 0.95, "tags": ["регион","география","регион название"]},

    # sales_dim
    {"dataset_id": 16, "schema": "dm", "table": "sales_dim", "column": "channel", "dtype": "string", "completeness": 0.99, "uniqueness": 0.95, "tags": ["канал","канал продаж","розница","marketplace"]},

    # customer_dim
    {"dataset_id": 17, "schema": "dm", "table": "customer_dim", "column": "customer_id", "dtype": "string", "completeness": 0.99, "uniqueness": 1.00, "tags": ["клиент","customer","ид клиента"]},

    # raw
    {"dataset_id": 12, "schema": "raw", "table": "erp_costs", "column": "cogs", "dtype": "decimal", "completeness": 0.92, "uniqueness": 0.90, "tags": ["себестоимость","cogs","затраты"]},
    {"dataset_id": 12, "schema": "raw", "table": "erp_costs", "column": "sku", "dtype": "string", "completeness": 0.94, "uniqueness": 0.80, "tags": ["sku","товар","артикул"]},

    {"dataset_id": 14, "schema": "raw", "table": "crm_events", "column": "event_type", "dtype": "string", "completeness": 0.91, "uniqueness": 0.65, "tags": ["событие","email","push","кампания"]},
])

# Build ref and reverse indexes
dataset_fields["ref"] = dataset_fields["schema"] + "." + dataset_fields["table"] + "." + dataset_fields["column"]
ref_to_quality = {r["ref"]: (r["completeness"], r["uniqueness"]) for _, r in dataset_fields.iterrows()}
ref_to_layer = {r["ref"]: ("vitrine" if r["schema"]=="dm" else "source") for _, r in dataset_fields.iterrows()}
ref_to_dataset = {r["ref"]: r["dataset_id"] for _, r in dataset_fields.iterrows()}

dataset_id_to_name = {r.dataset_id: r.name for _, r in datasets.iterrows()}
dataset_id_to_layer = {r.dataset_id: r.layer for _, r in datasets.iterrows()}

# Simple business glossary (terms → candidate fields/refs)
glossary = [
    {"term": "выручка", "syn": ["доход","оборот","revenue","sales"], "refs": ["dm.sales_facts.revenue","dm.finance_facts.revenue_actual","dm.finance_facts.revenue_budget"]},
    {"term": "маржа", "syn": ["прибыль","margin"], "refs": ["dm.sales_facts.price","raw.erp_costs.cogs"]},
    {"term": "себестоимость", "syn": ["cogs","затраты"], "refs": ["raw.erp_costs.cogs"]},
    {"term": "канал продаж", "syn": ["канал","розница","онлайн","marketplace","ecom"], "refs": ["dm.sales_facts.channel","dm.sales_dim.channel"]},
    {"term": "регион", "syn": ["гео","область","регион название"], "refs": ["dm.geo_dim.region_name","dm.sales_facts.region_id"]},
    {"term": "клиент", "syn": ["customer","ид клиента"], "refs": ["dm.customer_dim.customer_id"]},
    {"term": "отток", "syn": ["churn","ушёл","retention"], "refs": ["dm.customer_facts.churn_flag"]},
    {"term": "последняя покупка", "syn": ["recency","lrp"], "refs": ["dm.customer_facts.last_purchase_dt"]},
    {"term": "sku", "syn": ["товар","артикул","позиция"], "refs": ["dm.sales_facts.sku_id","raw.erp_costs.sku"]},
    {"term": "план дохода", "syn": ["бюджет","budget"], "refs": ["dm.finance_facts.revenue_budget"]},
]

stop_words = set("и или в на по за от до при как для к из у же же-то о об обо про над под между без около про".split())

# --------------------------- UTILS / LOGIC -----------------------------------
def normalize(q: str) -> list:
    q = (q or "").lower()
    tokens = []
    for t in q.replace(",", " ").replace(".", " ").replace("/", " ").replace("-", " ").split():
        if t and t not in stop_words:
            tokens.append(t)
    return tokens

def jaccard(a:set, b:set)->float:
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def match_glossary(query: str, top_k: int = 8):
    tokens = set(normalize(query))
    candidates = []
    for g in glossary:
        keyset = set([g["term"]] + g["syn"])
        score = jaccard(tokens, set(normalize(" ".join(list(keyset)))))
        if score>0:
            candidates.append({"term": g["term"], "score": score, "refs": g["refs"]})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    # Expand to field suggestions
    ref_scores = defaultdict(float)
    for c in candidates:
        for r in c["refs"]:
            ref_scores[r] = max(ref_scores[r], c["score"])
    # tag-based soft match across dataset_fields
    for _, row in dataset_fields.iterrows():
        tagset = set(normalize(" ".join(row["tags"])))
        s = jaccard(tokens, tagset)
        if s>0:
            ref_scores[row["ref"]] = max(ref_scores[row["ref"]], s*0.9)  # a bit lower than glossary
    ranked = sorted(ref_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return ranked  # list of (ref, score)

def feasibility_score(found_refs, total_refs, allow_vitrine=True):
    if total_refs == 0: 
        return 0, {}
    coverage = len(found_refs) / total_refs
    freshness = 0.8 if allow_vitrine else 0.9
    if found_refs:
        qualities = [ref_to_quality.get(r, (0.7,0.7))[0] for r in found_refs]
        quality = float(np.mean(qualities))
    else:
        quality = 0.5
    access = 0.9
    if found_refs:
        in_vitrine = [r for r in found_refs if ref_to_layer.get(r)=="vitrine"]
        reuse = 1.0 if len(in_vitrine) / len(found_refs) >= 0.7 else 0.4
    else:
        reuse = 0.3
    score = (
        coverage * 0.40 +
        freshness * 0.20 +
        quality * 0.15 +
        access * 0.15 +
        reuse * 0.10
    ) * 100
    breakdown = {
        "Покрытие полей": round(coverage*100),
        "Свежесть/SLA": round(freshness*100),
        "Качество данных": round(quality*100),
        "Доступ": round(access*100),
        "Переиспользование": round(reuse*100),
    }
    return round(score), breakdown

def status_label(score):
    if score >= 85: return "✅ Готово"
    if score >= 60: return "🟡 Частично"
    return "🔴 Требуется доработка"

def graphviz_lineage(refs:list):
    import graphviz as gv
    dot = gv.Digraph(format="svg")
    dot.attr(rankdir="LR")
    # Draw nodes for datasets and fields
    seen_ds = set()
    for r in refs:
        ds_id = ref_to_dataset.get(r)
        ds_name = dataset_id_to_name.get(ds_id, "dataset")
        layer = dataset_id_to_layer.get(ds_id, "source")
        ds_label = f"{ds_name}\n({layer})"
        if ds_id not in seen_ds:
            dot.node(f"ds_{ds_id}", ds_label, shape="folder" if layer=="vitrine" else "box3d")
            seen_ds.add(ds_id)
        dot.node(f"f_{r}", r.split(".")[-1], shape="note")
        dot.edge(f"ds_{ds_id}", f"f_{r}")
    # Simple pipeline chain demo
    for r in refs:
        if r.startswith("raw."):
            # show transform to dm.* if similarly named exists
            raw_tail = r.split(".",2)[-1]
            for r2 in refs:
                if r2.startswith("dm.") and r2.endswith(raw_tail.split(".")[-1]):
                    dot.edge(f"f_{r}", f"f_{r2}", label="transform")
    return dot

# ------------------------------ SIDEBAR --------------------------------------
st.sidebar.header("Глобальный поиск")
q = st.sidebar.text_input("Например: «хочу сравнить выручку и маржу по каналам и регионам за месяц»")
allow_vitrine = st.sidebar.toggle("Разрешить использовать витрину", value=True)
if st.sidebar.button("Подобрать поля"):
    ranked = match_glossary(q, top_k=10)
    st.sidebar.write("Подходящие поля:")
    for ref, sc in ranked:
        layer = ref_to_layer.get(ref, "?")
        st.sidebar.write(f"- {ref}  · {layer}  · score={sc:.2f}")

st.sidebar.markdown("---")
st.sidebar.header("Быстрая проверка (по списку полей)")
raw_req = st.sidebar.text_area("Список `schema.table.column` (по одному на строке)", "dm.sales_facts.revenue\ndm.sales_dim.channel\ndm.geo_dim.region_name\nraw.erp_costs.cogs")
if st.sidebar.button("Оценить пригодность"):
    requested = [x.strip() for x in raw_req.splitlines() if x.strip()]
    known = set(dataset_fields["ref"].tolist())
    found = [r for r in requested if r in known]
    score, br = feasibility_score(found, len(requested), allow_vitrine=allow_vitrine)
    st.sidebar.metric("Feasibility", f"{score}/100")
    st.sidebar.write(status_label(score))
    st.sidebar.write("Найдено:", found if found else "—")
    miss = [r for r in requested if r not in found]
    st.sidebar.write("Нет в каталоге:", miss if miss else "—")

# ------------------------------ HEADER ---------------------------------------
st.title("Каталог данных и отчётов → Подбор по хотелкам и проверка автоматизации")
st.caption("Определи: можно ли собрать новый отчёт из текущих данных и автоматизировать ручной.")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Подбор по хотелке", "📊 Отчёты", "🧱 Данные", "🧪 Конструктор проверки"])

# ------------------------------ TAB 1: WIZARD --------------------------------
with tab1:
    st.subheader("Опиши, что хочешь увидеть")
    example = "Нужна динамика выручки и маржи по каналам продаж и регионам за квартал."
    want = st.text_area("Свободный ввод (естественный язык)", value=example, height=90)
    gran = st.selectbox("Желаемая периодичность", ["день","неделя","месяц","квартал"], index=2)
    level = st.multiselect("Гранулярность (измерения)", ["регион","канал продаж","SKU","клиент"], default=["регион","канал продаж"])
    if st.button("Подобрать данные"):
        ranked = match_glossary(want, top_k=12)
        if not ranked:
            st.warning("Не нашла явных совпадений. Попробуй переформулировать запрос более предметно (выручка/клиент/канал/регион).")
        else:
            # Build suggestion table
            rows = []
            for ref, sc in ranked:
                ds_id = ref_to_dataset.get(ref)
                ds = datasets[datasets.dataset_id==ds_id].iloc[0]
                rows.append({
                    "Поле": ref.split(".")[-1],
                    "Источник": ref,
                    "Набор": ds.name,
                    "Слой": ds.layer,
                    "SLA (мин)": ds.sla_minutes,
                    "PII": ds.pii_flags or "—",
                    "Качество набора": ds.quality_score,
                    "Score": round(sc, 2)
                })
            st.write("**Кандидаты полей под вашу хотелку:**")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Choose top 5 distinct by semantic diversity (by dataset preference to vitrine)
            picked = []
            seen_cols = set()
            for ref, sc in ranked:
                col = ref.split(".")[-1]
                if col in seen_cols: 
                    continue
                seen_cols.add(col)
                picked.append(ref)
                if len(picked)>=5: break

            st.markdown("### Предложенный минимальный набор полей")
            st.code("\n".join(picked))

            # Feasibility using picked
            score, br = feasibility_score(picked, len(picked), allow_vitrine=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Feasibility", f"{score}/100")
            c2.metric("Статус", status_label(score))
            c3.metric("Покрытие полей", f"{br.get('Покрытие полей',0)}%")
            st.progress(min(1.0, score/100))
            with st.expander("Детализация оценки"):
                st.write(br)

            # Reuse suggestions: existing automated reports overlapping
            st.markdown("### Переиспользование существующих отчётов/витрины")
            # overlap with reports
            rep_overlap = []
            for rid, group in report_fields.groupby("report_id"):
                fields = set(group["source_ref"].tolist())
                inter = set(picked) & fields
                if inter:
                    ratio = len(inter)/len(picked)
                    rep = reports[reports.report_id==rid].iloc[0]
                    rep_overlap.append({
                        "Отчёт": rep["name"],
                        "Авто?": "Да" if rep["is_automated"] else "Нет",
                        "Совпадение": f"{int(ratio*100)}%",
                        "Совпавшие поля": ", ".join([r.split(".")[-1] for r in inter])
                    })
            if rep_overlap:
                st.dataframe(pd.DataFrame(rep_overlap), use_container_width=True)
            else:
                st.info("Прямого переиспользования отчётов не найдено. Сосредоточься на витрине и датасетах ниже.")

            # Reuse via vitrine share
            vit_share = sum(1 for r in picked if ref_to_layer.get(r)=="vitrine")/max(1,len(picked))
            if vit_share >= 0.7:
                st.success("≥70% полей доступны в витрине (dm.*) — рекомендуем переиспользовать её напрямую.")
            else:
                st.info("Существенная часть полей в RAW — потребуется пайплайн в витрину/датамарт.")

            # Lineage graph
            st.markdown("### Линия данных (черновик)")
            try:
                dot = graphviz_lineage(picked)
                st.graphviz_chart(dot)
            except Exception as e:
                st.caption(f"Графвиз недоступен: {e}")

            # Action plan
            st.markdown("### План автоматизации (черновик)")
            steps = []
            if vit_share < 0.7:
                steps.append("Вынести расчёты и ключевые поля в витрину (dm.*), выровнять гранулярность под требуемую периодичность.")
            if "клиент" in level and any("PII" in (datasets[datasets.dataset_id==ref_to_dataset[r]].iloc[0].pii_flags or "") for r in picked):
                steps.append("Проверить PII/доступы для клиентских полей, настроить маски и роли.")
            steps.append("Сравнить SLA набора с требуемой частотой и обновить расписание если нужно.")
            steps.append("Зафиксировать бизнес-термины в глоссарии (выручка/маржа/канал/регион).")
            for i,s in enumerate(steps,1):
                st.write(f"{i}. {s}")

# ------------------------------ TAB 2: REPORTS -------------------------------
with tab2:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего отчётов", len(reports))
    col2.metric("Автоматизировано", int(reports["is_automated"].sum()))
    col3.metric("Витрин", datasets.query("layer=='vitrine'").shape[0])
    col4.metric("Источников", datasets.query("layer!='vitrine'").shape[0])

    st.write("### Каталог отчётов")
    show = reports.copy()
    st.dataframe(show[["name","owner","business_domain","frequency","is_automated","automation_score","description"]]
                 .rename(columns={"name":"Название","owner":"Владелец","business_domain":"Домен",
                                  "frequency":"Частота","is_automated":"Авто?","automation_score":"Скор","description":"Описание"}),
                 use_container_width=True, height=240)

    st.markdown("### Детали")
    selected = st.selectbox("Выбери отчёт", options=show["name"].tolist())
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

    with st.expander("Рекомендации по переиспользованию"):
        in_vitrine_share = (rf["is_from_vitrine"].mean() if not rf.empty else 0)
        if in_vitrine_share >= 0.7:
            st.success("≥70% полей из витрины — высока вероятность быстрого расширения/переиспользования.")
        else:
            st.info("Заметная доля полей из RAW — стоит вынести их в датамарт и стабилизировать расчёты.")

# ------------------------------ TAB 3: DATA ----------------------------------
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Витрина (dm.*)**")
        st.dataframe(datasets.query("layer=='vitrine'")[["name","owner","sla_minutes","pii_flags","quality_score","granularity"]]
                     .rename(columns={"name":"Набор","owner":"Владелец","sla_minutes":"SLA (мин)","pii_flags":"PII","quality_score":"Качество","granularity":"Гранулярность"}),
                     use_container_width=True, height=240)
    with c2:
        st.write("**Источники (RAW/Source)**")
        st.dataframe(datasets.query("layer!='vitrine'")[["name","owner","sla_minutes","pii_flags","quality_score","granularity"]]
                     .rename(columns={"name":"Набор","owner":"Владелец","sla_minutes":"SLA (мин)","pii_flags":"PII","quality_score":"Качество","granularity":"Гранулярность"}),
                     use_container_width=True, height=240)
    st.markdown("### Поля")
    st.dataframe(dataset_fields[["schema","table","column","dtype","completeness","uniqueness","tags"]]
                 .rename(columns={"schema":"Схема","table":"Таблица","column":"Поле","dtype":"Тип","completeness":"Полнота","uniqueness":"Уникальность","tags":"Теги"}),
                 use_container_width=True, height=300)

# ------------------------------ TAB 4: FEASIBILITY ---------------------------
with tab4:
    st.write("Вставь список полей (`schema.table.column`) или собери из каталога.")
    req = st.text_area("Требуемые поля", "dm.sales_facts.revenue\ndm.sales_dim.channel\ndm.geo_dim.region_name\ndm.customer_facts.churn_flag")
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
        if found:
            vit_share = sum(1 for r in found if ref_to_layer.get(r)=="vitrine")/len(found)
            if vit_share >= 0.7:
                st.success("Рекомендуем переиспользовать витрину: ≥70% полей доступны в `dm.*`.")
            else:
                st.info("Часть полей только в источниках — добавьте их в датамарт/витрину.")
        st.markdown("#### План действий (черновик)")
        steps = []
        if miss:
            steps.append("Добавить отсутствующие поля в пайплайн (RAW → CLEAN → Витрина), описать трансформации.")
        if found:
            vit_share = sum(1 for r in found if ref_to_layer.get(r)=="vitrine")/len(found)
            if vit_share < 0.7:
                steps.append("Укрепить витрину: вынести расчёты в датамарт, выровнять гранулярность.")
        steps.append("Проверить доступы и флаги PII, синхронизировать SLA с требуемой частотой.")
        for i, s in enumerate(steps, 1):
            st.write(f"{i}. {s}")
