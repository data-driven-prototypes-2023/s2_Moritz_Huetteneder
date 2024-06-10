import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import joblib
import openai
from openai import OpenAI
import os


# Load the combined CSV file
beverages_df = pd.read_csv('Assignment 2/data/beverages_combined.csv')
print("DataFrame loaded successfully")
print(beverages_df.info())

# Load the pre-trained model
model = joblib.load('Assignment 2/nutriscore_decision_tree_model.pkl')
# Filter data to include only valid Nutriscore grades
valid_grades = ['a', 'b', 'c', 'd', 'e']
beverages_df_filtered = beverages_df[beverages_df['Nutriscore Grade'].isin(valid_grades)]

# Create visualizations
fig_nutriscore = px.bar(
    beverages_df_filtered, 
    x='Nutriscore Grade', 
    title='Distribution of Nutriscore Grades',
    color='Nutriscore Grade',
    color_discrete_map={
        'a': '#4caf50',  # Green
        'b': '#8bc34a',  # Light Green
        'c': '#ffeb3b',  # Yellow
        'd': '#ffc107',  # Amber
        'e': '#f44336'   # Red
    },
    hover_data=['Categories', 'Brands']
)

fig_nutriscore.update_layout(
    xaxis_title='Nutriscore Grade',
    yaxis_title='Count',
    title_x=0.5,
    template='plotly_white'
)

fig_ecoscore = px.bar(
    beverages_df, 
    x='Ecoscore Grade', 
    title='Distribution of Ecoscore Grades',
    color='Ecoscore Grade',
    color_discrete_map={
        'a': '#4caf50',  # Green
        'b': '#8bc34a',  # Light Green
        'c': '#ffeb3b',  # Yellow
        'd': '#ffc107',  # Amber
        'e': '#f44336'   # Red
    },
    hover_data=['Categories', 'Brands']
)

fig_ecoscore.update_layout(
    xaxis_title='Ecoscore Grade',
    yaxis_title='Count',
    title_x=0.5,
    template='plotly_white'
)
# Convert non-numeric values to NaN for relevant columns
numeric_columns = ['Sugars (g)', 'Proteins (g)', 'Fat (g)', 'Saturated Fat (g)', 'Salt (g)', 'Energy (kcal)']
beverages_df[numeric_columns] = beverages_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values in these columns
beverages_df_cleaned = beverages_df.dropna(subset=numeric_columns)

# Calculate average nutritional values by country
avg_nutritional_values = beverages_df_cleaned.groupby('Countries')[numeric_columns].mean().reset_index()

# Create additional visualizations
fig_heatmap = px.imshow(
    avg_nutritional_values.set_index('Countries').T,
    labels=dict(x='Countries', y='Nutritional Metrics', color='Average Value'),
    title='Heatmap of Average Nutritional Values by Country'
)

fig_stacked_bar = px.bar(
    avg_nutritional_values.melt(id_vars='Countries', value_vars=numeric_columns),
    x='Countries',
    y='value',
    color='variable',
    title='Stacked Bar Chart of Nutritional Composition by Country'
)

fig_sunburst = px.sunburst(
    beverages_df,
    path=['Countries', 'Categories'],
    values='Energy (kcal)',
    title='Sunburst Chart for Food Categories by Country'
)
# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H2("Beverage Analysis Dashboard"),
        html.Hr(),
        html.Div([
            dcc.Link('Country Comparison Analysis', href='/country-comparison'),
            html.Br(),
            dcc.Link('Individual Product Analysis', href='/product-analysis'),
            html.Br(),
            dcc.Link('AI Insights', href='/ai-insights')
        ], style={'padding': '20px'}),
    ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),
    
    html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
])
# Define the callback to update the content based on the current URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/country-comparison':
        return html.Div([
            html.H2("Country Comparison Analysis"),
            dcc.Graph(id='nutriscore-bar-chart', figure=fig_nutriscore),
            dcc.Graph(id='ecoscore-bar-chart', figure=fig_ecoscore),
            dcc.Graph(id='heatmap', figure=fig_heatmap),
            dcc.Graph(id='stacked-bar', figure=fig_stacked_bar),
            dcc.Graph(id='sunburst', figure=fig_sunburst),
            html.Label("Select Nutritional Metrics:"),
            dcc.Dropdown(
                id='nutrition-metrics-dropdown',
                options=[{'label': col, 'value': col} for col in numeric_columns],
                value=numeric_columns[0],
                multi=False
            ),
            dcc.Graph(id='nutrition-boxplot')
        ])
    elif pathname == '/product-analysis':
        return html.Div([
            html.H2("Individual Product Analysis"),
            html.Div([
                dcc.Dropdown(
                    id='product-dropdown',
                    options=[{'label': prod, 'value': prod} for prod in beverages_df['Product Code'].unique()],
                    placeholder='Select a product',
                    multi=True
                ),
                dcc.Graph(id='product-nutriscore-bar'),
                dcc.Graph(id='product-ecoscore-pie'),
                dcc.Graph(id='product-nutrition-radar')
            ]),
            html.H3("Predict Nutriscore for New Product"),
            html.Div([
                html.Label("Sugars (g):"),
                dcc.Input(id='input-sugars', type='number', value=0),
                html.Br(),
                html.Label("Proteins (g):"),
                dcc.Input(id='input-proteins', type='number', value=0),
                html.Br(),
                html.Label("Fat (g):"),
                dcc.Input(id='input-fat', type='number', value=0),
                html.Br(),
                html.Label("Saturated Fat (g):"),
                dcc.Input(id='input-saturated-fat', type='number', value=0),
                html.Br(),
                html.Label("Salt (g):"),
                dcc.Input(id='input-salt', type='number', value=0),
                html.Br(),
                html.Label("Energy (kcal):"),
                dcc.Input(id='input-energy', type='number', value=0),
                html.Br(),
                html.Button('Predict Nutriscore', id='predict-button', style={'marginTop': '10px'}),
                html.Div(id='prediction-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'})
        ])
    elif pathname == '/ai-insights':
        return html.Div([
            html.H2("AI Insights"),
            html.Div([
                dcc.Textarea(
                    id='ai-input',
                    placeholder='Enter your query here...',
                    style={'width': '100%', 'height': 100}
                ),
                html.Button('Get Insights', id='ai-button', style={'marginTop': '10px'}),
                html.Div(id='ai-output', style={'marginTop': '20px', 'fontWeight': 'bold'})
            ], style={'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'boxShadow': '0 0 10px rgba(0,0,0,0.1)'})
        ])
    else:
        return html.Div([html.H2("Welcome to the Beverage Analysis Dashboard")])
# Callback to update the nutrition box plot based on selected metric
@app.callback(
    Output('nutrition-boxplot', 'figure'),
    [Input('nutrition-metrics-dropdown', 'value')]
)
def update_nutrition_boxplot(selected_metric):
    if not selected_metric:
        selected_metric = numeric_columns[0]
    fig_boxplot = px.box(
        beverages_df_cleaned, 
        x='Countries', 
        y=selected_metric,
        title=f'Distribution of {selected_metric} by Country',
        points=False
    )
    fig_boxplot.update_layout(
        xaxis_title='Countries',
        yaxis_title=selected_metric,
        title_x=0.5,
        template='plotly_white'
    )
    return fig_boxplot
# Callback to update the product details graph based on selected product
@app.callback(
    [Output('product-nutriscore-bar', 'figure'),
     Output('product-ecoscore-pie', 'figure'),
     Output('product-nutrition-radar', 'figure')],
    [Input('product-dropdown', 'value')]
)
def update_product_analysis(selected_products):
    if not selected_products:
        return {}, {}, {}

    filtered_data = beverages_df[beverages_df['Product Code'].isin(selected_products)]

    fig_nutriscore_bar = px.bar(
        filtered_data,
        x='Product Code',
        y='Nutriscore Score',
        color='Nutriscore Grade',
        title='Nutriscore Distribution for Selected Products'
    )

    fig_ecoscore_pie = px.pie(
        filtered_data,
        names='Ecoscore Grade',
        title='Ecoscore Distribution for Selected Products'
    )

    fig_nutrition_radar = px.line_polar(
        filtered_data.melt(id_vars=['Product Code'], value_vars=numeric_columns),
        r='value',
        theta='variable',
        color='Product Code',
        line_close=True,
        title='Nutritional Composition Radar Chart for Selected Products'
    )

    return fig_nutriscore_bar, fig_ecoscore_pie, fig_nutrition_radar
# Callback to predict Nutriscore for new product
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [
        State('input-sugars', 'value'),
        State('input-proteins', 'value'),
        State('input-fat', 'value'),
        State('input-saturated-fat', 'value'),
        State('input-salt', 'value'),
        State('input-energy', 'value')
    ]
)
def predict_nutriscore(n_clicks, sugars, proteins, fat, saturated_fat, salt, energy):
    if n_clicks is not None:
        # Create a DataFrame for the input features
        input_data = pd.DataFrame({
            'Sugars (g)': [sugars],
            'Proteins (g)': [proteins],
            'Fat (g)': [fat],
            'Saturated Fat (g)': [saturated_fat],
            'Salt (g)': [salt],
            'Energy (kcal)': [energy]
        })
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        
        return f"The predicted Nutriscore is: {prediction}"
    return ""

# Initialize OpenAI client
api_key = '' # Add your OpenAI API key here
client = OpenAI(api_key=api_key)

# Callback to get AI insights
@app.callback(
    Output('ai-output', 'children'),
    [Input('ai-button', 'n_clicks')],
    [State('ai-input', 'value')]
)
def get_ai_insights(n_clicks, query):
    if n_clicks is not None and query:
        response = client.completions.create(
            model="text-davinci-003",
            prompt=query,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    return ""

# Run the app with Flask development server
if __name__ == '__main__':
    app.run_server(debug=True)

