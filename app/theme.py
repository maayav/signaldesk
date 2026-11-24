"""Small visual system shared by the local Streamlit workspace."""

import streamlit as st


COLORS = {
    "mint": "#78E4C2",
    "amber": "#F2C06B",
    "blue": "#89B9FF",
    "muted": "#9CAEC4",
    "panel": "#132237",
    "background": "#0B1524",
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #0b1524; color: #edf3fa; }
        html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
        [data-testid="stSidebar"] { background: #0e1b2d; border-right: 1px solid #213249; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
        .block-container { padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1450px; }
        h1, h2, h3 { font-family: 'Inter', 'Segoe UI', sans-serif !important; letter-spacing: -.035em; }
        h1 { font-weight: 800 !important; }
        p, li { line-height: 1.6; }
        [data-testid="stMetric"] { background: #132237; border: 1px solid #25364d; border-radius: 12px; padding: 18px 20px; }
        [data-testid="stMetricLabel"] { color: #9caec4; font-size: .83rem; }
        [data-testid="stMetricValue"] { font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 1.8rem; }
        [data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #25364d !important; border-radius: 12px; }
        .stTextArea textarea, .stTextInput input { background: #101e31; color: #edf3fa; border: 1px solid #314158; border-radius: 10px; }
        .stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 600; }
        .stButton > button[kind="primary"] { background: #78e4c2; color: #10261f; border: 1px solid #78e4c2; }
        .stButton > button[kind="secondary"] { color: #dce7f4; background: #16263b; border: 1px solid #314158; }
        .stTabs [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid #25364d; }
        .stTabs [aria-selected="true"] { color: #78e4c2 !important; }
        .sd-brand { font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 1.5rem; font-weight: 800; letter-spacing: -.05em; color: #edf3fa; margin-bottom: 3px; }
        .sd-brand span { color: #78e4c2; }
        .sd-overline { font-size: .68rem; font-weight: 700; letter-spacing: .16em; color: #78e4c2; text-transform: uppercase; margin-bottom: 10px; }
        .sd-subtitle { color: #9caec4; max-width: 750px; margin-top: -9px; margin-bottom: 25px; font-size: 1rem; }
        .sd-sidebar-caption { color: #8195ae; font-size: .78rem; margin-bottom: 30px; }
        .sd-pill { display: inline-block; border-radius: 100px; font-size: .72rem; font-weight: 600; padding: 5px 11px; border: 1px solid #35544f; background: #16382f; color: #8ce8cd; }
        .sd-result { border: 1px solid #35544f; background: linear-gradient(125deg, #15382f, #132237); border-radius: 14px; padding: 24px 26px; margin: 12px 0 20px; }
        .sd-result.review { border-color: #6d5734; background: linear-gradient(125deg, #352e22, #132237); }
        .sd-result .label { color: #9caec4; font-size: .72rem; text-transform: uppercase; letter-spacing: .12em; margin-bottom: 9px; }
        .sd-result .intent { font-family: 'Inter', 'Segoe UI', sans-serif; color: #edf3fa; font-weight: 800; font-size: 1.5rem; letter-spacing: -.035em; }
        .sd-result .detail { color: #b6c5d8; font-size: .85rem; margin-top: 8px; }
        .sd-step { color: #8195ae; font-size: .73rem; letter-spacing: .03em; padding: 16px 0; border-top: 1px solid #25364d; }
        .sd-step span { color: #78e4c2; }
        .sd-footnote { color: #8195ae; font-size: .75rem; line-height: 1.6; }
        [data-testid="stCaptionContainer"] { color: #94a8c0; }
        header[data-testid="stHeader"] { background: rgba(11,21,36,.85); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, *, height: int = 300):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#B6C5D8", "size": 12},
        margin={"l": 0, "r": 15, "t": 15, "b": 0},
        height=height,
        legend={"orientation": "h", "y": 1.15, "x": 0},
    )
    fig.update_xaxes(gridcolor="#223249", zerolinecolor="#223249")
    fig.update_yaxes(gridcolor="#223249", zerolinecolor="#223249")
    return fig
