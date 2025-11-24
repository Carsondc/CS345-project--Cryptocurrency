from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Your HTML file

@app.route("/submit", methods=["POST"])
def submit():
    selected = request.form.get("selectedItems")   # From textarea/hidden input
    days = request.form.get("days")                # From number input

    print("Selected cryptos:", selected)
    print("Number of days:", days)

    return f"""
        <h2>Received Data</h2>
        <p><strong>Cryptos:</strong> {selected}</p>
        <p><strong>Days:</strong> {days}</p>
    """, days, selected
if __name__ == '__main__':
    # Use debug=True for development to enable reloader and debugger
    app.run(debug=True)
