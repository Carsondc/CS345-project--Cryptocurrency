from flask import Flask, request, render_template

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

        return render_template("index.html",
                            cryptos=selected,
                            days=days)
    

if __name__ == '__main__':
    app.run(debug=True)

