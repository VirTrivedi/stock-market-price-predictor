from flask import Flask, render_template, request
from model import predict_price  # Import the predict_price function from model.py
from data_fetch import get_stock_data

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        future_steps = int(request.form['future_steps'])
        
        # Get the predicted price and the explanation
        predicted_price, explanation = predict_price(ticker, future_steps)
        
        # Get the current price and predicted price change from the model
        df = get_stock_data(ticker)  # You might want to import or define this function
        current_price = df['Close'].iloc[-1]
        predicted_price_change = predicted_price - current_price
        
        return render_template('index.html', ticker=ticker, 
                               future_steps=future_steps, 
                               predicted_price=predicted_price,
                               current_price=current_price,
                               predicted_price_change=predicted_price_change,
                               explanation=explanation)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)