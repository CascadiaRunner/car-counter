import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import psycopg2
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration - use DATABASE_URL from Heroku
DATABASE_URL = os.environ.get('DATABASE_URL')

def get_daily_counts():
    """Query the database for daily car counts"""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # Query to get daily counts
    query = """
        SELECT 
            DATE(timestamp) as date,
            SUM(count) as total_cars
        FROM car_counts
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT 30
    """
    
    cur.execute(query)
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    dates = [str(row[0]) for row in results]
    counts = [row[1] for row in results]
    return dates, counts

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# For Heroku deployment
server = app.server

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Daily Car Count Dashboard", className="text-center my-4"))
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='daily-counts-graph'),
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ])
    ])
])

@app.callback(
    dash.Output('daily-counts-graph', 'figure'),
    dash.Input('interval-component', 'n_intervals')
)
def update_graph(n):
    dates, counts = get_daily_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=dates,
            y=counts,
            marker_color='rgb(55, 83, 109)'
        )
    ])
    
    fig.update_layout(
        title='Daily Car Counts',
        xaxis_title="Date",
        yaxis_title="Number of Cars",
        showlegend=False
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True) 