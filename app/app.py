import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
from io import BytesIO
import base64

# ---------------------------
# Load model
# ---------------------------
xgb_model = joblib.load("xgb_model_ai4i.joblib")  # Your trained XGBoost model
FEATURES = ['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']
explainer = shap.TreeExplainer(xgb_model)

# ---------------------------
# Initialize Dash
# ---------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Predictive Maintenance Dashboard"

# ---------------------------
# Layout
# ---------------------------
app.layout = html.Div([
    html.H1("PREDICTIVE MAINTENANCE SYSTEM", 
            style={'textAlign':'center', 'marginBottom':'20px','color':'#222'}),

    dcc.Tabs(id="tabs", value='tab1', children=[
        dcc.Tab(label='Predictive Maintenance', value='tab1'),
        dcc.Tab(label='Feature Explainability', value='tab2'),
        dcc.Tab(label='Historical Trends', value='tab3'),
        dcc.Tab(label='Recommendations', value='tab4')
    ]),

    html.Div(id='tabs-content'),
    dcc.Store(id='latest-shap', data=None)
], style={'padding':'20px','minHeight':'100vh'})

# ---------------------------
# Render Tab Content
# ---------------------------
@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab1':
        return html.Div([
            html.H3("Enter Sensor Values", style={'color':'#333'}),
            html.Div([
                html.Div([
                    html.Label(f'{f.replace("_"," ")}', style={'fontWeight':'bold'}),
                    dcc.Input(id=f.lower(), type='number', value=0, step=0.1, style={'width':'100%'})
                ], style={'marginBottom':'10px','width':'48%'}) for f in FEATURES
            ], style={'display':'flex','flexWrap':'wrap','justifyContent':'space-between'}),

            html.Br(),
            html.Div(id='prediction-output', 
                     style={'marginTop':'20px','fontSize':'18px','fontWeight':'bold','color':'#222'}),

            dcc.Graph(id='risk-gauge', style={'height':'350px'})
        ], style={'backgroundColor':'rgba(255,255,255,0.5)','padding':'20px','borderRadius':'10px'})

    elif tab == 'tab2':
        return html.Div([
            html.H3("Feature Contributions (SHAP)", style={'color':'#333'}),
            dcc.Graph(id='shap-bar', style={'height':'450px'})
        ], style={'backgroundColor':'rgba(255,255,255,0.5)','padding':'20px','borderRadius':'10px'})

    elif tab == 'tab3':
        return html.Div([
            html.H3("Upload Historical Sensor Data", style={'color':'#333'}),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select CSV')]),
                style={'width':'60%','height':'60px','lineHeight':'60px',
                       'borderWidth':'1px','borderStyle':'dashed','borderRadius':'5px',
                       'textAlign':'center','margin':'10px auto','backgroundColor':'white'},
                multiple=False
            ),
            dcc.Graph(id='historical-trends', style={'height':'400px'})
        ], style={'backgroundColor':'rgba(255,255,255,0.5)','padding':'20px','borderRadius':'10px'})

    elif tab == 'tab4':
        return html.Div([
            html.H3("Maintenance Recommendations", style={'color':'#333'}),
            html.Ul([
                html.Li("If risk > 50% → Schedule immediate maintenance."),
                html.Li("If risk 20–50% → Monitor closely and plan maintenance."),
                html.Li("If risk < 20% → Continue normal operation."),
                html.Li("Check SHAP bars to identify major contributors.")
            ], style={'fontSize':'16px'})
        ], style={'backgroundColor':'rgba(255,255,255,0.5)','padding':'20px','borderRadius':'10px'})


# ---------------------------
# Real-time Prediction Callback
# ---------------------------
@app.callback(
    Output('prediction-output', 'children'),
    Output('risk-gauge', 'figure'),
    Output('latest-shap', 'data'),
    [Input(f.lower(), 'value') for f in FEATURES]
)
def realtime_prediction(*values):
    # Convert all inputs to float
    values = [float(v) if v is not None else 0 for v in values]

    df = pd.DataFrame([values], columns=FEATURES)

    # If all inputs are 0 → baseline
    if all(v == 0 for v in values):
        pred_text = "Waiting for input — all sensor values are 0."
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            title={'text': "Risk Score"},
            gauge={'axis': {'range':[0,100]},
                   'bar': {'color': "lightgray"},
                   'steps':[{'range':[0,20],'color':'lightgreen'},
                            {'range':[20,50],'color':'yellow'},
                            {'range':[50,100],'color':'red'}]}
        ))
        shap_data = {'x':[0]*len(FEATURES), 'y':FEATURES, 'vals':values}
        return pred_text, fig_gauge, shap_data

    # Predict from model
    pred = xgb_model.predict(df)[0]
    prob = xgb_model.predict_proba(df)
    risk_score = float(prob[0][1]*100)

    # Risk text
    if risk_score > 50:
        pred_text = f"High Risk: {risk_score:.2f}% → Failure Expected"
    elif risk_score > 20:
        pred_text = f"Medium Risk: {risk_score:.2f}% → Possible Failure"
    else:
        pred_text = f"Low Risk: {risk_score:.2f}% → No Failure Expected"

    # Gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Risk Score"},
        gauge={'axis': {'range':[0,100]},
               'bar': {'color': "red" if risk_score>50 else "green"},
               'steps':[{'range':[0,20],'color':'lightgreen'},
                        {'range':[20,50],'color':'yellow'},
                        {'range':[50,100],'color':'red'}]}
    ))

    # SHAP values
    try:
        shap_values = explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if shap_values.ndim > 1:
            shap_values = shap_values[0]
    except:
        shap_values = np.zeros(len(FEATURES))

    shap_data = {'x': shap_values.tolist(), 'y': FEATURES, 'vals': values}

    return pred_text, fig_gauge, shap_data


# ---------------------------
# Update SHAP Plot
# ---------------------------
@app.callback(
    Output('shap-bar', 'figure'),
    Input('latest-shap', 'data')
)
def update_shap(shap_data):
    if shap_data is None:
        return go.Figure()

    colors = ['#2ca02c' if val > 0 else '#d62728' for val in shap_data['x']]
    text_vals = [f"{v}" for v in shap_data['vals']]

    abs_idx = np.argsort(np.abs(shap_data['x']))[::-1]
    x_sorted = np.array(shap_data['x'])[abs_idx]
    y_sorted = np.array(shap_data['y'])[abs_idx]
    colors_sorted = np.array(colors)[abs_idx]
    text_sorted = np.array(text_vals)[abs_idx]

    fig = go.Figure(go.Bar(
        x=x_sorted,
        y=y_sorted,
        orientation='h',
        marker_color=colors_sorted,
        text=text_sorted,
        textposition='outside'
    ))

    fig.update_layout(
        title="Feature Contributions (SHAP)",
        yaxis={'autorange':'reversed'},
        xaxis_title="SHAP Value (Impact on Prediction)",
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# ---------------------------
# Historical Trends Callback
# ---------------------------
@app.callback(
    Output('historical-trends', 'figure'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_historical(content, filename):
    if content is None:
        return go.Figure()

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(BytesIO(decoded))
    df.columns = df.columns.str.strip().str.replace(' ','_')

    fig = go.Figure()
    for feature in FEATURES:
        if feature in df.columns:
            fig.add_trace(go.Scatter(y=df[feature], mode='lines', name=feature))
    fig.update_layout(title="Historical Sensor Trends", xaxis_title="Time", yaxis_title="Value")
    return fig


# ---------------------------
# Run App
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True)
