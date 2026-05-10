"""
Returns S&P 500 stocks + a broad list of ETFs as the search universe.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ── Popular ETFs across all major categories ──────────────────────────────────
ETF_TICKERS = [
    # Broad market
    "SPY", "VOO", "IVV", "VTI", "QQQ", "DIA", "IWM", "IWB", "MDY", "IJR",
    # Sectors (SPDR)
    "XLF", "XLK", "XLV", "XLE", "XLI", "XLU", "XLB", "XLP", "XLY", "XLRE", "XLC",
    # Sectors (Vanguard)
    "VFH", "VGT", "VHT", "VDE", "VIS", "VPU", "VAW", "VDC", "VCR", "VGSIX",
    # Tech / themes
    "SMH", "SOXX", "HACK", "CLOU", "SKYY", "ARKK", "ARKG", "ARKF", "ARKW", "ARKQ",
    "BOTZ", "ROBO", "AIQ", "FINX", "WCLD",
    # Crypto / digital assets
    "IBIT", "GBTC", "BITO", "BITX", "ARKB", "FBTC",
    # Commodities
    "GLD", "IAU", "SLV", "GDX", "GDXJ", "USO", "UNG", "PDBC", "DBA", "CORN",
    "WEAT", "SOYB", "CPER", "PALL", "PPLT",
    # Fixed income
    "TLT", "IEF", "SHY", "BND", "AGG", "HYG", "LQD", "EMB", "BKLN", "JEPI",
    "JEPQ", "TIP", "VTIP", "FLOT", "VCSH",
    # International
    "EFA", "VEA", "EEM", "VWO", "FXI", "EWJ", "EWZ", "EWY", "INDA", "VGK",
    "EWG", "EWC", "EWA", "RSX", "MCHI",
    # Real estate
    "VNQ", "IYR", "SCHH", "RWR", "XLRE",
    # Leveraged / inverse (for pattern matching, not investment)
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SOXL", "SOXS", "TNA", "TZA", "LABU", "LABD",
    # Dividend / factor
    "SCHD", "VIG", "DVY", "HDV", "DGRO", "NOBL", "QUAL", "MTUM", "VLUE", "SIZE",
    # Multi-asset / balanced
    "AOM", "AOR", "AOA", "AOK", "VBIAX",
]


def get_all_tickers() -> list[str]:
    """Return S&P 500 stocks + ETFs."""
    stocks = _get_sp500()
    all_tickers = list(set(stocks + ETF_TICKERS))
    logger.info(f"Universe: {len(stocks)} S&P 500 stocks + {len(ETF_TICKERS)} ETFs = {len(all_tickers)} total")
    return all_tickers


def _get_sp500() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for tbl in tables:
            for col in tbl.columns:
                if col.lower() in ("symbol", "ticker"):
                    tickers = (
                        tbl[col].str.replace(".", "-", regex=False).dropna().tolist()
                    )
                    tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
                    if len(tickers) >= 490:
                        logger.info(f"S&P 500 from Wikipedia: {len(tickers)} tickers")
                        return tickers
    except Exception as exc:
        logger.warning(f"Wikipedia fetch failed: {exc}")
    return _hardcoded_sp500()


def _hardcoded_sp500() -> list[str]:
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","BRK-B","AVGO",
        "JPM","LLY","UNH","V","XOM","MA","COST","HD","PG","WMT","JNJ","NFLX",
        "ABBV","BAC","CRM","ORCL","CVX","MRK","KO","AMD","PEP","TMO","ACN","MCD",
        "CSCO","ABT","GE","IBM","DHR","AXP","QCOM","GS","MS","ISRG","TXN","NEE",
        "INTU","CAT","RTX","BKNG","SPGI","BLK","PFE","T","VZ","MDT","AMGN","GILD",
        "HON","AMAT","SYK","ETN","LOW","ELV","CB","VRTX","LMT","MMC","TJX","ADP",
        "PLD","SBUX","MU","COF","REGN","BMY","CL","CME","SO","DUK","ICE","BSX",
        "PANW","WM","FCX","SHW","MCO","LRCX","ITW","HCA","NOC","GD","AON",
        "PH","EMR","USB","ECL","APH","TT","NSC","WELL","CTAS","WMB","CCI","TDG",
        "AFL","NKE","FDX","PSA","A","FICO","GWW","CARR","FAST","EW","MSI",
        "KLAC","ADI","MCHP","CDNS","SNPS","FTNT","CRWD","WDAY","NOW","DDOG","ZS",
        "TEAM","ABNB","DASH","TTD","APP","PLTR","ARM",
        "WFC","PNC","TFC","MTB","KEY","RF","CFG","HBAN","FITB","STT","BK","SCHW",
        "AMP","PRU","MET","ALL","TRV","PGR","HIG",
        "CVS","CI","HUM","MOH","CNC","MCK","CAH","DGX","LH",
        "PM","MO","CHRW","EXPD","ODFL","CSX","UNP","DAL","UAL","AAL","LUV",
        "SPG","AMT","EQIX","PSA","EXR","AVB","EQR","NVR","PHM","LEN","DHI","TOL",
        "KHC","GIS","CPB","SJM","CAG","HRL","MKC","TSN","HSY","MDLZ","MNST","KDP",
        "STZ","DD","DOW","LYB","PPG","APD","ALB","CE","EMN","FMC","IFF","MOS",
        "NUE","STLD","CLF","AA","PKG","IP","BALL","DE","AGCO","ROK","IR","XYL",
        "FTV","GNRC","SWK","SNA","AME","ROP","PNR","OTIS","TXT","HII","LHX",
        "LDOS","SAIC","BAH","HPE","HPQ","DELL","STX","WDC","NTAP","PSTG","TER",
        "ON","SWKS","MPWR","ZBH","BAX","BDX","COO","HOLX","IDXX","IQV","PODD",
        "TFX","ALGN","DXCM","INSP","EOG","COP","DVN","FANG","HAL","SLB","BKR",
        "OXY","VLO","MPC","PSX","D","EXC","AEP","SRE","PEG","PPL","FE","EIX",
        "ES","ETR","CNP","AEE","CMS","NI","WEC","DTE","LNT","EVRG","AWK","XEL",
        "ED","ATO","UGI","RXO","ALK","CNH","IQV","PPG","TDG","EMN","MO",
    ]
