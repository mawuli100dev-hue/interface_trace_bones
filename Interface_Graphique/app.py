"""
app.py — Wildlife Classifier v11
- Suppression de l'option dossier
- Aperçu image au clic sur le nom de fichier dans le tableau
"""

import io, os, tempfile
import pandas as pd
import streamlit as st
from PIL import Image

from engine import (
    classify_image, compute_stats,
    launch_training, load_model, results_to_dataframe, SUPPORTED_EXTENSIONS,
)

st.set_page_config(page_title="Wildlife Classifier", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
:root {
  --bg:#f7f7f6; --surface:#ffffff; --green-lt:#f0faf3;
  --border-act:#d1e7d6; --ink:#111210; --muted:#6b6b68;
  --faint:#b5b5b2; --green:#1b5e35; --green-dk:#14472a;
  --red:#b91c1c; --radius:8px; --font:'Inter',sans-serif;
}
html,body,[class*="css"]{font-family:var(--font)!important;background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border-act)!important;}
[data-testid="stSidebar"] *{color:var(--ink)!important;}
[data-baseweb="tab-list"]{background:var(--surface)!important;border-bottom:1px solid var(--border-act)!important;}
[data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-size:.8rem!important;font-weight:500!important;padding:10px 26px!important;border-bottom:2px solid transparent!important;transition:color .15s,border-color .15s!important;}
[aria-selected="true"][data-baseweb="tab"]{color:var(--green)!important;border-bottom-color:var(--green)!important;background:var(--green-lt)!important;}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{background:var(--surface)!important;border:1px solid var(--border-act)!important;color:var(--ink)!important;border-radius:var(--radius)!important;}
[data-testid="stButton"]>button{background:var(--surface)!important;border:1px solid var(--border-act)!important;color:var(--ink)!important;border-radius:var(--radius)!important;font-size:.82rem!important;font-weight:500!important;transition:all .15s!important;}
[data-testid="stButton"]>button:hover{background:var(--green-lt)!important;border-color:var(--green)!important;color:var(--green)!important;}
button[kind="primary"]{background:var(--green)!important;border-color:var(--green)!important;color:#fff!important;font-weight:600!important;}
button[kind="primary"]:hover{background:var(--green-dk)!important;border-color:var(--green-dk)!important;}
[data-testid="stDownloadButton"]>button{background:var(--surface)!important;border:1px solid var(--border-act)!important;color:var(--green)!important;border-radius:var(--radius)!important;font-size:.82rem!important;font-weight:500!important;}
[data-testid="stDownloadButton"]>button:hover{background:var(--green-lt)!important;border-color:var(--green)!important;}
[data-testid="stFileUploader"]{background:var(--surface)!important;border:1.5px dashed var(--border-act)!important;border-radius:var(--radius)!important;}
[data-testid="stFileUploader"]:hover{background:var(--green-lt)!important;border-color:var(--green)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--border-act)!important;border-radius:var(--radius)!important;overflow:hidden;background:var(--surface)!important;}
[data-testid="stVegaLiteChart"]{background:var(--surface)!important;border:1px solid var(--border-act)!important;border-radius:var(--radius)!important;padding:8px!important;}
[data-testid="stRadio"] label{color:var(--ink)!important;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "model"            not in st.session_state: st.session_state.model            = None
if "class_names"      not in st.session_state: st.session_state.class_names      = []
if "model_name"       not in st.session_state: st.session_state.model_name       = ""
if "results"          not in st.session_state: st.session_state.results          = {}
if "last_single"      not in st.session_state: st.session_state.last_single      = None
if "train_log"        not in st.session_state: st.session_state.train_log        = ""
if "train_running"    not in st.session_state: st.session_state.train_running    = False
if "train_progress"   not in st.session_state: st.session_state.train_progress   = 0.0
if "prev_single_name" not in st.session_state: st.session_state.prev_single_name = None
if "prev_multi_names" not in st.session_state: st.session_state.prev_multi_names = set()
# Nom du fichier sélectionné pour l'aperçu dans le tableau
if "preview_file"     not in st.session_state: st.session_state.preview_file     = None

def add_result(row: dict):
    st.session_state.results[row["Fichier"]] = row

def all_results() -> list[dict]:
    return list(st.session_state.results.values())

# ── Couleurs ──────────────────────────────────────────────────────────────────
GREEN   = "#1b5e35"
GREEN_L = "#f0faf3"
BORDER  = "#d1e7d6"
INK     = "#111210"
MUTED   = "#b5b5b2"
RED     = "#b91c1c"
GREY    = "#6b6b68"

def section(text):
    st.markdown(
        f"<p style='font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;"
        f"color:#b5b5b2;font-weight:600;margin:22px 0 8px;"
        f"padding-bottom:6px;border-bottom:1px solid {BORDER};'>{text}</p>",
        unsafe_allow_html=True)

def stat_card(value, title, color=GREEN, sub=None):
    sub_html = f"<div style='font-size:.63rem;color:#b5b5b2;margin-top:3px;'>{sub}</div>" if sub else ""
    return (f"<div style='background:#fff;border:1px solid {BORDER};"
            f"border-top:2px solid {color};border-radius:8px;"
            f"padding:14px 12px 10px;text-align:center;'>"
            f"<div style='font-size:.58rem;letter-spacing:.1em;text-transform:uppercase;"
            f"color:#b5b5b2;font-weight:600;margin-bottom:5px;'>{title}</div>"
            f"<div style='font-size:1.85rem;font-weight:600;color:{color};"
            f"line-height:1;font-variant-numeric:tabular-nums;'>{value}</div>"
            f"{sub_html}</div>")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding-bottom:18px;margin-bottom:4px;border-bottom:1px solid {BORDER};">
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:4px;height:28px;background:{GREEN};border-radius:2px;flex-shrink:0;"></div>
        <div>
          <div style="font-size:.95rem;font-weight:600;color:{INK};">Wildlife Classifier</div>
          <div style="font-size:.67rem;color:{GREEN};margin-top:1px;letter-spacing:.06em;">ResNet</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    section("Modèle")
    model_file = st.file_uploader("h5", type=["h5"], label_visibility="collapsed",
                                  help="Limite → .streamlit/config.toml → maxUploadSize")
    if model_file:
        if model_file.name != st.session_state.model_name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
                tmp.write(model_file.read()); tmp_path = tmp.name
            with st.spinner("Chargement du modèle…"):
                mdl, cn = load_model(tmp_path)
            st.session_state.model            = mdl
            st.session_state.class_names      = cn
            st.session_state.model_name       = model_file.name
            st.session_state.results          = {}
            st.session_state.last_single      = None
            st.session_state.preview_file     = None
            st.session_state.prev_single_name = None
            st.session_state.prev_multi_names = set()
            st.success(f"✓ {model_file.name}")

    if st.session_state.model_name:
        st.markdown(
            f"<p style='font-size:.7rem;color:#b5b5b2;margin-top:6px;line-height:1.7;'>"
            + " · ".join(st.session_state.class_names) + "</p>",
            unsafe_allow_html=True)

    section("Seuil de confiance")
    threshold_pct = st.slider("s", 50, 99, 70, 1,
                              label_visibility="collapsed", format="%d%%")
    threshold = threshold_pct / 100
    st.markdown(
        f"<p style='font-size:.7rem;color:#b5b5b2;margin-top:-4px;'>"
        f"En dessous de <b style='color:{INK};'>{threshold_pct}%</b> → <i>inconnu</i></p>",
        unsafe_allow_html=True)

    res_list = all_results()
    if res_list:
        st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:18px 0;'>",
                    unsafe_allow_html=True)
        df_side = results_to_dataframe(res_list)
        st.download_button("⬇  Exporter CSV", data=df_side.to_csv(index=False).encode(),
                           file_name="resultats.csv", mime="text/csv",
                           key="dl_sidebar", use_container_width=True)
        if st.button("↺  Réinitialiser", use_container_width=True):
            st.session_state.results          = {}
            st.session_state.last_single      = None
            st.session_state.preview_file     = None
            st.session_state.prev_single_name = None
            st.session_state.prev_multi_names = set()
            st.rerun()

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2 = st.tabs(["  Classification  ", "  Réentraînement  "])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFICATION
# ════════════════════════════════════════════════════════════════════════════
with t1:
    st.markdown(f"<h1 style='font-size:1.15rem;font-weight:600;color:{INK};"
                f"margin:18px 0 14px;'>Classification</h1>", unsafe_allow_html=True)

    if not st.session_state.model:
        st.markdown(
            f"<div style='border-left:3px solid {GREEN};padding:11px 16px;"
            f"background:{GREEN_L};border-radius:0 8px 8px 0;font-size:.83rem;"
            f"color:{GREEN};'>Chargez un modèle <b>.h5</b> dans la barre latérale.</div>",
            unsafe_allow_html=True)
        st.stop()

    model       = st.session_state.model
    class_names = st.session_state.class_names

    # ── Deux colonnes d'entrée (dossier supprimé) ──────────────────────────
    ca, cb = st.columns(2, gap="large")

    with ca:
        section("Image unique")
        up1 = st.file_uploader("s", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                               key="s1", label_visibility="collapsed")
    with cb:
        section("Images multiples")
        upm = st.file_uploader("m", type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
                               accept_multiple_files=True, key="sm",
                               label_visibility="collapsed")

    # ── Image unique ───────────────────────────────────────────────────────
    if up1 is not None:
        if up1.name != st.session_state.prev_single_name:
            raw = up1.read()
            ext = os.path.splitext(up1.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(raw); tp = tmp.name
            pred, conf = classify_image(model, class_names, tp)
            if conf < threshold: pred = "inconnu"
            add_result({"Fichier": up1.name, "Classe": pred,
                        "Confiance": f"{conf*100:.2f}%",
                        "_conf_raw": conf, "_raw_bytes": raw})
            st.session_state.prev_single_name = up1.name
            st.session_state.last_single      = up1.name

        last = st.session_state.last_single
        if last and last in st.session_state.results:
            row  = st.session_state.results[last]
            pred = row["Classe"]
            conf = row["_conf_raw"]
            section("Aperçu")
            c1, c2 = st.columns([1, 2], gap="medium")
            with c1:
                img = Image.open(io.BytesIO(row["_raw_bytes"]))
                img.thumbnail((240, 240))
                st.image(img)
            with c2:
                unk  = pred in ("inconnu", "Erreur")
                col  = GREY if unk else GREEN
                bg   = "#f5f5f4" if unk else GREEN_L
                bd   = "#ebebea" if unk else BORDER
                icon = "—" if unk else "✓"
                pct_bar = int(conf * 100)
                st.markdown(
                    f"<div style='margin-top:16px;background:{bg};"
                    f"border:1px solid {bd};border-radius:8px;padding:16px 18px;'>"
                    f"<div style='font-size:.62rem;letter-spacing:.1em;"
                    f"text-transform:uppercase;color:#b5b5b2;margin-bottom:6px;'>Prédiction</div>"
                    f"<div style='font-size:1.5rem;font-weight:600;color:{col};"
                    f"margin-bottom:12px;'>{icon}&nbsp;{pred}</div>"
                    f"<div style='background:{BORDER};border-radius:999px;height:4px;overflow:hidden;'>"
                    f"<div style='height:4px;border-radius:999px;background:{col};"
                    f"width:{pct_bar}%;'></div></div>"
                    f"<div style='font-size:.7rem;color:#b5b5b2;margin-top:7px;'>"
                    f"Confiance : <b style='color:{INK};'>{conf*100:.2f}%</b></div>"
                    f"</div>", unsafe_allow_html=True)

    # ── Images multiples ───────────────────────────────────────────────────
    if upm:
        current_names = {uf.name for uf in upm}
        nouveaux = [uf for uf in upm
                    if uf.name not in st.session_state.prev_multi_names]
        if nouveaux:
            with st.spinner(f"Analyse de {len(nouveaux)} nouvelle(s) image(s)…"):
                for uf in nouveaux:
                    ext = os.path.splitext(uf.name)[1] or ".jpg"
                    raw = uf.read()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(raw); p = tmp.name
                    try:
                        pred, conf = classify_image(model, class_names, p)
                        if conf < threshold: pred = "inconnu"
                        row = {"Fichier": uf.name, "Classe": pred,
                               "Confiance": f"{conf*100:.2f}%",
                               "_conf_raw": conf,
                               "_raw_bytes": raw}   # ← on stocke les bytes pour l'aperçu
                    except Exception:
                        row = {"Fichier": uf.name, "Classe": "Erreur",
                               "Confiance": "0%", "_conf_raw": 0.0,
                               "_raw_bytes": None}
                    add_result(row)
            st.session_state.prev_multi_names |= current_names

    # ── Stats & tableau ────────────────────────────────────────────────────
    res_list = all_results()
    if res_list:
        stats = compute_stats(res_list)
        total = len(res_list)
        n_unk = stats.get("inconnu", 0)
        n_err = stats.get("Erreur",  0)
        n_ok  = total - n_unk - n_err

        cards = [("Total", total, INK), ("Identifiés", n_ok, GREEN)]
        if n_unk > 0: cards.append(("Inconnus", n_unk, GREY))
        if n_err > 0: cards.append(("Erreurs",  n_err, RED))
        for cls, cnt in sorted(stats.items()):
            if cls in ("inconnu", "Erreur"): continue
            cards.append((cls.capitalize(), cnt, GREEN))

        section("Résumé")
        cols = st.columns(len(cards), gap="small")
        for i, (ttl, val, col) in enumerate(cards):
            with cols[i]:
                pct_s = f"{round(val/total*100)}%" if total else "—"
                st.markdown(stat_card(val, ttl, col, pct_s), unsafe_allow_html=True)

        # ── Tableau avec sélection pour aperçu ────────────────────────────
        section("Résultats  —  cliquez sur une ligne pour voir la photo")

        df = results_to_dataframe(res_list)

        # st.dataframe avec on_select pour détecter la ligne cliquée
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=min(540, 56 + len(df) * 35),
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "Fichier":   st.column_config.TextColumn("Fichier",   width="large"),
                "Classe":    st.column_config.TextColumn("Classe",    width="medium"),
                "Confiance": st.column_config.TextColumn("Confiance", width="small"),
            },
        )

        # Récupération de la ligne sélectionnée
        selected_rows = event.selection.rows if event.selection else []
        if selected_rows:
            idx = selected_rows[0]
            fname = df.iloc[idx]["Fichier"]
            st.session_state.preview_file = fname

        # ── Panneau d'aperçu ──────────────────────────────────────────────
        pf = st.session_state.preview_file
        if pf and pf in st.session_state.results:
            row = st.session_state.results[pf]
            raw_bytes = row.get("_raw_bytes")
            if raw_bytes:
                pred = row["Classe"]
                conf = row["_conf_raw"]
                unk  = pred in ("inconnu", "Erreur")
                col  = GREY if unk else GREEN
                bg   = "#f5f5f4" if unk else GREEN_L
                bd   = "#ebebea" if unk else BORDER
                icon = "—" if unk else "✓"

                st.markdown(
                    f"<div style='margin-top:16px;background:{bg};"
                    f"border:1px solid {bd};border-radius:10px;"
                    f"padding:16px 20px;display:flex;align-items:flex-start;gap:20px;'>",
                    unsafe_allow_html=True)

                pc1, pc2 = st.columns([1, 2], gap="medium")
                with pc1:
                    img = Image.open(io.BytesIO(raw_bytes))
                    img.thumbnail((280, 280))
                    st.image(img, caption=pf)
                with pc2:
                    st.markdown(
                        f"<div style='padding-top:8px;'>"
                        f"<div style='font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;"
                        f"color:#b5b5b2;margin-bottom:4px;'>Fichier</div>"
                        f"<div style='font-size:.9rem;font-weight:600;color:{INK};"
                        f"margin-bottom:14px;word-break:break-all;'>{pf}</div>"
                        f"<div style='font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;"
                        f"color:#b5b5b2;margin-bottom:4px;'>Prédiction</div>"
                        f"<div style='font-size:1.4rem;font-weight:700;color:{col};"
                        f"margin-bottom:14px;'>{icon}&nbsp;{pred}</div>"
                        f"<div style='background:{BORDER};border-radius:999px;"
                        f"height:5px;overflow:hidden;width:100%;'>"
                        f"<div style='height:5px;border-radius:999px;background:{col};"
                        f"width:{int(conf*100)}%;'></div></div>"
                        f"<div style='font-size:.72rem;color:#b5b5b2;margin-top:6px;'>"
                        f"Confiance : <b style='color:{INK};'>{conf*100:.2f}%</b></div>"
                        f"</div>",
                        unsafe_allow_html=True)

        st.download_button(
            "⬇  Télécharger les résultats (CSV)",
            data=df.to_csv(index=False).encode(),
            file_name="resultats_wildlife.csv",
            mime="text/csv",
            key="dl_table",
        )

        anm = {k: v for k, v in stats.items() if k != "Erreur"}
        if anm:
            section("Distribution")
            cdf = pd.DataFrame(
                [(k.capitalize(), v) for k, v in sorted(anm.items(), key=lambda x: -x[1])],
                columns=["Classe", "Nombre"])
            st.bar_chart(cdf.set_index("Classe"), color=GREEN,
                         height=180, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RÉENTRAÎNEMENT
# ════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown(
        f"<h1 style='font-size:1.15rem;font-weight:600;color:{INK};"
        f"margin:18px 0 4px;'>Réentraînement</h1>"
        f"<p style='font-size:.78rem;color:#b5b5b2;margin-bottom:16px;'>"
        f"Exécute <code style='background:{GREEN_L};padding:1px 6px;"
        f"border-radius:4px;color:{GREEN};border:1px solid {BORDER};'>training.py</code>"
        f" en arrière-plan.</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Répertoires")
        train_dir = st.text_input("train", placeholder="/data/train",
                                  label_visibility="collapsed")
        val_dir   = st.text_input("val",   placeholder="/data/val",
                                  label_visibility="collapsed")
        save_path = st.text_input("out",   placeholder="/models/model_v2.h5",
                                  label_visibility="collapsed")
    with c2:
        section("Hyperparamètres")
        epochs = st.number_input("Époques", min_value=1, max_value=2000,
                                 value=100, step=5)
        device = st.radio("Périphérique", ["GPU", "CPU"], horizontal=True)

    st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:20px 0;'>",
                unsafe_allow_html=True)

    go = st.button(" Lancer l'entraînement",
                   disabled=st.session_state.train_running, type="primary")

    if go:
        if not all([train_dir, val_dir, save_path]):
            st.error("Remplissez tous les champs.")
        elif not os.path.isdir(train_dir):
            st.error("Dossier d'entraînement introuvable.")
        elif not os.path.isdir(val_dir):
            st.error("Dossier de validation introuvable.")
        else:
            st.session_state.update(train_running=True, train_log="", train_progress=0.0)
            lines: list[str] = []
            def _ln(l):  lines.append(l); st.session_state.train_log = "".join(lines)
            def _pr(p):  st.session_state.train_progress = p
            def _dn(rc): st.session_state.train_running = False
            launch_training(train_dir, val_dir, save_path, epochs,
                            device.lower(), _ln, _pr, _dn)
            st.success("Entraînement démarré ✓")

    if st.session_state.train_running or st.session_state.train_log:
        section("Progression")
        pct = int(st.session_state.train_progress)
        st.markdown(
            f"<div style='background:{BORDER};border-radius:999px;height:5px;"
            f"overflow:hidden;margin-bottom:6px;'>"
            f"<div style='height:5px;border-radius:999px;background:{GREEN};"
            f"width:{pct}%;transition:width .4s;'></div></div>"
            f"<div style='font-size:.7rem;color:#b5b5b2;margin-bottom:10px;'>{pct}%</div>",
            unsafe_allow_html=True)
        log = st.session_state.train_log or "En attente…"
        st.markdown(
            f"<div style='background:{GREEN_L};border:1px solid {BORDER};"
            f"border-radius:8px;padding:14px 18px;font-family:monospace;"
            f"font-size:.74rem;color:{GREEN};max-height:280px;"
            f"overflow-y:auto;white-space:pre-wrap;line-height:1.7;'>{log}</div>",
            unsafe_allow_html=True)
        if st.session_state.train_running:
            if st.button(" Rafraîchir"): st.rerun()
        else:
            st.success(" Entraînement terminé.")
