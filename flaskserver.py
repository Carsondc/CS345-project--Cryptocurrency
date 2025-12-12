from flask import Flask, request, render_template
import subprocess
import os

app = Flask(__name__)
class flaskserver:
    selected = []
    days = 0

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/submit", methods=["POST"])
    def submit():
        selected = request.form.get("selectedItems")
        days = request.form.get("days")

        print("Selected cryptos:", selected)
        print("Number of days:", days)
        currency_to_file = {
        "1inch Network": "1INCH.csv",
        "Aave": "AAVE.csv",
        "Cardano": "ADA.csv",
        "Algorand": "ALGO.csv",
        "Amp": "AMP.csv",
        "ApeCoin": "APE.csv",
        "Arweave": "AR.csv",
        "Cosmos": "ATOM.csv",
        "Avalanche": "AVAX.csv",
        "Axie Infinity": "AXS.csv",
        "Basic Attention Token": "BAT.csv",
        "Bitcoin Cash": "BCH.csv",
        "BNB": "BNB.csv",
        "Bitcoin SV": "BSV.csv",
        "Bitcoin": "BTC.csv",
        "BitTorrent": "BTT.csv",
        "PancakeSwap": "CAKE.csv",
        "Conflux Network": "CFX.csv",
        "Chiliz": "CHZ.csv",
        "Compound": "COMP.csv",
        "Cronos": "CRO.csv",
        "Curve DAO Token": "CRV.csv",
        "Convex Finance": "CVX.csv",
        "Dai": "DAI.csv",
        "Dash": "DASH.csv",
        "Decred": "DCR.csv",
        "DigiByte": "DGB.csv",
        "Dogecoin": "DOGE.csv",
        "Polkadot": "DOT.csv",
        "dYdX": "DYDX.csv",
        "MultiversX (formerly Elrond)": "EGLD.csv",
        "Ethereum Name Service": "ENS.csv",
        "Ethereum Classic": "ETC.csv",
        "Ethereum": "ETH.csv",
        "Fetch.ai": "FET.csv",
        "Filecoin": "FIL.csv",
        "Flow": "FLOW.csv",
        "FTX Token": "FTT.csv",
        "Gala": "GALA.csv",
        "Golem": "GLM.csv",
        "The Graph": "GRT.csv",
        "GateToken": "GT.csv",
        "Hedera": "HBAR.csv",
        "Helium": "HNT.csv",
        "Internet Computer": "ICP.csv",
        "Immutable X": "IMX.csv",
        "Injective Protocol": "INJ.csv",
        "IOTA": "IOTA.csv",
        "JUST": "JST.csv",
        "KuCoin Token": "KCS.csv",
        "Kusama": "KSM.csv",
        "Lido DAO": "LDO.csv",
        "UNUS SED LEO": "LEO.csv",
        "Chainlink": "LINK.csv",
        "Livepeer": "LPT.csv",
        "Litecoin": "LTC.csv",
        "Decentraland": "MANA.csv",
        "Mina Protocol": "MINA.csv",
        "MEXC Token": "MX.csv",
        "NEAR Protocol": "NEAR.csv",
        "NEO": "NEO.csv",
        "Nexo": "NEXO.csv",
        "APENFT": "NFT.csv",
        "OKB": "OKB.csv",
        "PAX Gold": "PAXG.csv",
        "Quant": "QNT.csv",
        "Qtum": "QTUM.csv",
        "Raydium": "RAY.csv",
        "Reserve Rights": "RSR.csv",
        "THORChain": "RUNE.csv",
        "The Sandbox": "SAND.csv",
        "Shiba Inu": "SHIB.csv",
        "Synthetix": "SNX.csv",
        "Solana": "SOL.csv",
        "Stacks": "STX.csv",
        "Sun Token": "SUN.csv",
        "SuperVerse": "SUPER.csv",
        "PancakeSwap Syrup": "SYRUP.csv",
        "Theta Network": "THETA.csv",
        "Trust Wallet Token": "TWT.csv",
        "Uniswap": "UNI.csv",
        "USD Coin": "USDC.csv",
        "USDD": "USDD.csv",
        "Tether": "USDT.csv",
        "VeChain": "VET.csv",
        "WEMIX": "WEMIX.csv",
        "Tether Gold": "XAUt.csv",
        "Onyxcoin (formerly Chain)": "XCN.csv",
        "eCash": "XEC.csv",
        "Stellar": "XLM.csv",
        "Monero": "XMR.csv",
        "NANO / XNO": "XNO.csv",
        "Ripple (XRP)": "XRP.csv",
        "Tezos": "XTZ.csv",
        "Zcash": "ZEC.csv",
        "Horizen": "ZEN.csv",
        "0x Protocol": "ZRX.csv",
    }
        files = []
        file = currency_to_file[selected]
        files.append(file)
        result = subprocess.run(
        ["python", "linear-regression.py", str(days), file],
        capture_output=True,
        text=True
    )
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PRED_PATH = os.path.join(BASE_DIR, "prediction.txt")

        with open(PRED_PATH) as f:
            prediction = f.read()


        return render_template(
            "index.html",
            prediction=prediction,
            currency=selected,
            days=days

        )
    
@app.route("/images")
def images():
    return render_template("images.html")

if __name__ == '__main__':
    app.run(debug=False)

